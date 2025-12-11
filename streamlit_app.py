import streamlit as st
import os
import json
import time
from google.cloud import storage, documentai_v1 as documentai
import vertexai
from vertexai.generative_models import GenerativeModel

# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(page_title="Contract Risk Scanner", layout="wide")
st.title("📜 Legal Contract Risk Scanner")
st.caption("Analyzes contracts using Document AI + Gemini and returns clause-level JSON with actionable recommendations.")

st.sidebar.title("⚙️ Settings")
PROJECT_ID = "contract-risk-scanner"
LOCATION = "us"
VERTEX_LOCATION = "us-central1"
BUCKET_NAME = "contract-risk-scanner-bucket-7119"
PROCESSOR_ID = "e2f1e97f3572e66"

# =========================
# STEP 1 — AUTHENTICATION (upload once)
# =========================
st.sidebar.header("🔐 Google Cloud Auth (upload once)")

if "auth_configured" not in st.session_state:
    st.session_state["auth_configured"] = False

if not st.session_state["auth_configured"]:
    uploaded_key = st.sidebar.file_uploader("Upload your key.json (only once)", type=["json"])
    if uploaded_key:
        key_path = "gcp_key.json"
        with open(key_path, "wb") as f:
            f.write(uploaded_key.getbuffer())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        st.session_state["auth_configured"] = True
        st.sidebar.success("✅ Google Cloud Authenticated (saved as gcp_key.json)")
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
    st.sidebar.success("✅ Using saved credentials")

# =========================
# STEP 2 — PDF UPLOAD
# =========================
def upload_to_gcs(file, filename):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    timestamp = int(time.time())
    blob_name = f"contracts/{timestamp}_{filename}"
    blob = bucket.blob(blob_name)
    file.seek(0)
    blob.upload_from_file(file)
    return f"gs://{BUCKET_NAME}/{blob_name}"

uploaded_pdf = st.file_uploader("📂 Upload Your Contract (PDF)", type=["pdf"])

if not uploaded_pdf:
    st.info("Upload a contract PDF to start.")
    st.stop()

# Upload to GCS
st.info("⏳ Uploading contract to Google Cloud Storage...")
gcs_path = upload_to_gcs(uploaded_pdf, uploaded_pdf.name)
st.success(f"✅ Uploaded to: {gcs_path}")

# =========================
# STEP 3 — DOCUMENT AI: extract text (not shown)
# =========================
st.info("🔍 Extracting text from contract using Document AI...")
docai_client = documentai.DocumentProcessorServiceClient()
processor_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
# blob_name is everything after gs://<bucket>/
blob_name = gcs_path.replace(f"gs://{BUCKET_NAME}/", "")
blob = bucket.blob(blob_name)
pdf_bytes = blob.download_as_bytes()

raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
result = docai_client.process_document(request)

if not (result.document and result.document.text):
    st.error("❌ Could not extract text from the document. Try a different file or processor.")
    st.stop()

extracted_text = result.document.text.strip()
# Save the extracted text for download (but DO NOT display)
with open("extracted_contract_text.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)
st.success("🎯 Text extraction completed successfully. Ready for risk analysis phase.")
st.download_button("📥 Download Extracted Text (optional)", extracted_text, file_name="extracted_contract_text.txt")

# =========================
# STEP 4 — GEMINI: risk analysis with actionable recommendations
# =========================
if st.button("🤖 Run Contract Risk Analysis"):
    with st.spinner("Analyzing contract using Gemini 2.5 Flash Lite..."):
        try:
            # Initialize Vertex AI
            vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
            model = GenerativeModel("gemini-2.5-flash-lite")

            # Prompt: requires clause list + actionable_recommendations + overall summary
            prompt = f"""
You are a legal contract analysis assistant. Analyze the contract text below and return a single VALID JSON object with two top-level keys:

1) "clauses": a JSON array where each item is an object with these exact fields:
   - clause_id: integer (sequential starting at 1)
   - original_text: the exact clause text (short paragraph)
   - simplified_text: one-line plain-English summary
   - risk_category: one of ["Termination", "Compensation", "Confidentiality", "Liability", "Non-compete", "Data Sharing", "Jurisdiction", "Auto-Renewal", "Penalty Fees", "Unilateral Changes", "Other"]
   - severity: one of ["High", "Medium", "Low"]
   - why_it_matters: one-sentence explanation of legal significance
   - actionable_recommendations: an array of 1-3 short, practical, prioritized next steps (e.g., "Negotiate clause X", "Seek counsel", "Request clarification", "Remove automatic renewal", "Limit penalty to X", etc.)

2) "actionable_recommendations_summary": an array of 3-6 high-level, prioritized actions covering the entire contract (for negotiation and caution). Each item must be short and specific (max 20 words).

Important rules:
- Respond ONLY with the JSON object (no preamble, no commentary).
- Ensure valid JSON; do not include markdown or code fences.
- Limit clause text to concise chunks (not full document). Produce up to 20 clauses if applicable.

Contract text (first 5000 characters):
{extracted_text[:5000]}
"""

            response = model.generate_content(prompt)
            analysis_raw = response.text.strip()

            # Try to parse JSON. If model appended extra text, attempt to slice the JSON.
            try:
                analysis_obj = json.loads(analysis_raw)
            except json.JSONDecodeError:
                # try to extract JSON object between first '{' and last '}'
                start = analysis_raw.find("{")
                end = analysis_raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        analysis_obj = json.loads(analysis_raw[start:end+1])
                    except Exception:
                        analysis_obj = None
                else:
                    analysis_obj = None

            if analysis_obj is None:
                st.warning("⚠️ Gemini output was not valid JSON. Showing raw output and saving as text.")
                st.text_area("Raw Gemini Output", analysis_raw, height=400)
                # Save raw fallback
                with open("contract_risk_analysis.txt", "w", encoding="utf-8") as f:
                    f.write(analysis_raw)
                st.download_button("📥 Download Raw Analysis (TXT)", analysis_raw, file_name="contract_risk_analysis.txt")
            else:
                # Pretty display and downloads
                st.success("✅ Legal Risk Analysis Completed")
                # Show JSON in an expandable viewer
                st.subheader("📋 Clause-level Analysis (JSON)")
                st.json(analysis_obj)

                # Save JSON file
                json_data = json.dumps(analysis_obj, indent=2, ensure_ascii=False)
                with open("contract_risk_analysis.json", "w", encoding="utf-8") as f:
                    f.write(json_data)

                st.download_button(
                    label="📥 Download Risk Analysis (JSON)",
                    data=json_data,
                    file_name="contract_risk_analysis.json",
                    mime="application/json"
                )

                # Additionally show the high-level actionable_recommendations_summary if present
                summary = analysis_obj.get("actionable_recommendations_summary")
                if summary:
                    st.subheader("🧭 High-level Actionable Recommendations")
                    for i, item in enumerate(summary, start=1):
                        st.markdown(f"**{i}.** {item}")

        except Exception as e:
            st.error(f"❌ AI Analysis failed: {e}")
