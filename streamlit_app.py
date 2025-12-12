# streamlit_app.py
import streamlit as st
import os
import json
import time
import re
from tempfile import NamedTemporaryFile

# Google auth & clients
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import documentai_v1 as documentai

# Vertex / Gemini
import vertexai
from vertexai.generative_models import GenerativeModel

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Contract Risk Scanner", layout="wide")
st.title("📜 Legal Contract Risk Scanner")
st.caption("Document AI → Gemini pipeline. Produces clause-level JSON with actionable recommendations.")

# -------------------------
# App config (edit if needed)
# -------------------------
APP_CONFIG = st.secrets.get("app", {})  # optional config block in secrets
PROJECT_ID = APP_CONFIG.get("project_id", "contract-risk-scanner")
LOCATION = APP_CONFIG.get("location", "us")                      # Document AI region
VERTEX_LOCATION = APP_CONFIG.get("vertex_location", "us-central1")
BUCKET_NAME = APP_CONFIG.get("bucket_name", "contract-risk-scanner-bucket-7119")
PROCESSOR_ID = APP_CONFIG.get("processor_id", "e2f1e97f3572e66")

# -------------------------
# Load credentials from Streamlit secrets
# -------------------------
if "google_service_account" not in st.secrets:
    st.error(
        "Google service account not found in Streamlit secrets.\n\n"
        "Add the full service account JSON to Streamlit secrets under the key: 'google_service_account'."
    )
    st.stop()

service_account_info = st.secrets["google_service_account"]

try:
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
except Exception as e:
    st.error(f"Failed to construct credentials from secrets: {e}")
    st.stop()

# -------------------------
# Initialize Google clients using credentials
# -------------------------
try:
    storage_client = storage.Client(credentials=credentials, project=PROJECT_ID)
    docai_client = documentai.DocumentProcessorServiceClient(credentials=credentials)
except Exception as e:
    st.error(f"Failed to initialize Google clients: {e}")
    st.stop()

# -------------------------
# Helper: sanitize and parse model output to JSON
# -------------------------
def sanitize_and_parse(analysis_raw: str):
    """Try parse JSON; attempt simple fixes if model output is malformed."""
    # direct parse
    try:
        return json.loads(analysis_raw)
    except Exception:
        pass

    text = analysis_raw.strip()

    # remove common numeric labels like 0:{ inside arrays
    text = re.sub(r'(\[)\s*\d+\s*:', r'\1', text)
    text = re.sub(r',\s*\d+\s*:', ',', text)
    text = re.sub(r'\{\s*\d+\s*:', '{', text)

    # convert single quotes to double quotes only if it looks like JSON with single quotes
    if "'" in text and '"' not in text[:200]:
        text = text.replace("'", '"')

    # extract first JSON object or array candidate
    start_obj = text.find('{')
    end_obj = text.rfind('}')
    start_arr = text.find('[')
    end_arr = text.rfind(']')

    candidate = None
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj:end_obj+1]
    elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = text[start_arr:end_arr+1]

    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None

# Upload helper
def upload_to_gcs_fileobj(file_obj, filename):
    bucket = storage_client.bucket(BUCKET_NAME)
    ts = int(time.time())
    safe_name = filename.replace(" ", "_")
    blob_name = f"contracts/{ts}_{safe_name}"
    blob = bucket.blob(blob_name)
    file_obj.seek(0)
    blob.upload_from_file(file_obj)
    return f"gs://{BUCKET_NAME}/{blob_name}", blob_name
# UI: file uploader
st.header("Upload contract PDF")
uploaded_pdf = st.file_uploader("📂 Upload Your Contract (PDF)", type=["pdf"])

if uploaded_pdf is None:
    st.info("Upload a PDF for analysis.")
    st.stop()

# Upload to GCS
with st.spinner("⏳ Uploading contract to Google Cloud Storage..."):
    try:
        gcs_uri, blob_name = upload_to_gcs_fileobj(uploaded_pdf, uploaded_pdf.name)
        st.success("✅ File Uploaded")
    except Exception as e:
        st.error(f"Upload failed: {e}")
        st.stop()
# Document AI: extract text (direct)
st.info("🔍 Extracting text from contract...")
processor_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

try:
    # download the uploaded object bytes back (or use uploaded_pdf.getvalue())
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    pdf_bytes = blob.download_as_bytes()

    raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
    result = docai_client.process_document(request)

    if not (result and getattr(result, "document", None)):
        st.error("Document AI returned no document object.")
        st.stop()

    extracted_text = (result.document.text or "").strip()
    if not extracted_text:
        st.error("Document AI extracted no text. Try another file or configure a different processor.")
        st.stop()

    # write extracted text to a file for optional download (not displayed)
    with open("extracted_contract_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    st.success("🎯 Text extraction completed successfully. Ready for risk analysis phase.")
    st.download_button("📥 Download Extracted Text (optional)", extracted_text, file_name="extracted_contract_text.txt")

except Exception as e:
    st.error(f"Document AI processing failed: {e}")
    st.stop()
# Gemini (Vertex) analysis
st.header("AI Risk Analysis")
if st.button("🤖 Run Contract Risk Analysis"):
    with st.spinner("Analyzing contract..."):
        try:
            # Vertex init: prefer passing credentials if supported, otherwise write temp key file
            try:
                vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION, credentials=credentials)
            except TypeError:
                # fallback: create a temp JSON key file and set env var (lives only in runtime)
                tmpf = NamedTemporaryFile(delete=False, suffix=".json")
                tmpf.write(json.dumps(service_account_info).encode("utf-8"))
                tmpf.flush()
                tmpf.close()
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmpf.name
                vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)

            model = GenerativeModel("gemini-2.5-flash-lite")

            prompt = f"""
You are a legal contract analysis assistant. Analyze the contract text below and return a SINGLE VALID JSON OBJECT with two top-level keys:

1) "clauses": an array where each item is an object with these exact fields:
   - clause_id (integer)
   - original_text (string)
   - simplified_text (string)
   - risk_category (one of ["Termination","Compensation","Confidentiality","Liability","Non-compete","Data Sharing","Jurisdiction","Auto-Renewal","Penalty Fees","Unilateral Changes","Other"])
   - severity (one of ["High","Medium","Low"])
   - why_it_matters (string)
   - actionable_recommendations (array of short strings, 1-3 items)

2) "actionable_recommendations_summary": array of 3-6 short, prioritized actions for the entire contract.

Important rules:
- Respond ONLY with a single valid JSON object. No commentary, no markdown, no numbered prefixes.
- Limit the clause extraction to concise chunks, up to 20 clauses.
- Keep each "simplified_text" to one sentence, and actionable recommendations to very short instructions.

Contract text (first 5000 characters):
{extracted_text[:5000]}
"""
            response = model.generate_content(prompt)
            analysis_raw = response.text.strip()

            # attempt to parse JSON; sanitize if necessary
            analysis_obj = sanitize_and_parse(analysis_raw)

            if analysis_obj is None:
                st.warning("⚠️ Gemini output could not be parsed as JSON. Showing raw output and providing TXT download.")
                st.text_area("Raw Gemini Output", analysis_raw, height=400)
                with open("contract_risk_analysis.txt", "w", encoding="utf-8") as f:
                    f.write(analysis_raw)
                st.download_button("📥 Download Raw Analysis (TXT)", analysis_raw, file_name="contract_risk_analysis.txt")
            else:
                pretty_json = json.dumps(analysis_obj, indent=2, ensure_ascii=False)
                with open("contract_risk_analysis.json", "w", encoding="utf-8") as f:
                    f.write(pretty_json)

                st.success("✅ Legal Risk Analysis Completed")
                st.subheader("📋 Clause-level Analysis (JSON)")
                st.json(analysis_obj)

                st.download_button(
                    label="📥 Download Risk Analysis (JSON)",
                    data=pretty_json,
                    file_name="contract_risk_analysis.json",
                    mime="application/json"
                )

                # show high-level actionable summary if present
                summary = analysis_obj.get("actionable_recommendations_summary")
                if isinstance(summary, list) and summary:
                    st.subheader("🧭 High-level Actionable Recommendations")
                    for i, item in enumerate(summary, start=1):
                        st.markdown(f"**{i}.** {item}")

        except Exception as e:
            st.error(f"AI analysis failed: {e}")
            st.exception(e)
