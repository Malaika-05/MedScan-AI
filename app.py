from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
CORS(app)


# ── Register routes directly here ─────────────────
import uuid
from flask import request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

from utils.yolo_infer import detect_regions, crop_region
from utils.ocr_utils  import extract_text_easyocr
from utils.bert_infer import classify_symptoms
from utils.rag_utils  import retrieve
from utils.llm_utils  import explain_report, explain_symptoms

UPLOAD_FOLDER = os.path.join("data", "uploads")
ALLOWED_EXTS  = {"png", "jpg", "jpeg", "pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.0"})


@app.route("/api/upload-report", methods=["POST"])
def upload_report():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    lang = request.form.get("language", "english")

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Step 1: YOLO
        detections = detect_regions(filepath)

        # Step 2: OCR
        findings_region = next(
            (d for d in detections if "finding" in d["class"].lower()
             or "report" in d["class"].lower()), None
        )

        if findings_region:
            cropped_img    = crop_region(filepath, findings_region["bbox"])
            extracted_text = extract_text_easyocr(cropped_img)
        else:
            extracted_text = extract_text_easyocr(Image.open(filepath))

        if not extracted_text.strip():
            extracted_text = "No readable text found in the report image."

        # Step 3: RAG
        rag_context = retrieve(extracted_text[:300], top_k=2)

        # Step 4: LLM
        explanation = explain_report(
            extracted_text=extracted_text,
            detected_regions=detections,
            patient_language=lang
        )

        return jsonify({
            "success":        True,
            "detections":     detections,
            "extracted_text": extracted_text[:500],
            "explanation":    explanation,
            "regions_found":  len(detections)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/check-symptoms", methods=["POST"])
def check_symptoms():
    data = request.get_json()

    if not data or "symptoms" not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms = data.get("symptoms", "").strip()
    lang     = data.get("language", "english")

    if len(symptoms) < 10:
        return jsonify({"error": "Please describe symptoms in more detail"}), 400

    try:
        # Step 1: BERT
        bert_result = classify_symptoms(symptoms)

        # Step 2: RAG
        rag_context = retrieve(f"{symptoms} {bert_result['category']}", top_k=3)

        # Step 3: LLM
        explanation = explain_symptoms(
            symptom_text=symptoms,
            bert_prediction=bert_result,
            rag_context=rag_context,
            patient_language=lang
        )

        return jsonify({
            "success":     True,
            "symptoms":    symptoms,
            "prediction":  bert_result,
            "explanation": explanation,
            "rag_used":    bool(rag_context)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)