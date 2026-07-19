import os
import uuid
from flask import request, jsonify, render_template, current_app as app
from werkzeug.utils import secure_filename
from PIL import Image

from utils.yolo_infer  import detect_regions, crop_region
from utils.ocr_utils   import extract_text_easyocr
from utils.bert_infer  import classify_symptoms
from utils.rag_utils   import retrieve
from utils.llm_utils   import explain_report, explain_symptoms

UPLOAD_FOLDER  = os.path.join("data", "uploads")
ALLOWED_EXTS   = {"png", "jpg", "jpeg", "pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ── Home page ─────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Health check ──────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.0"})


# ── Route 1: Upload medical report image ──────────
@app.route("/api/upload-report", methods=["POST"])
def upload_report():
    """
    Accepts a medical report image.
    Pipeline: YOLO → OCR → RAG → LLM
    Returns plain-language explanation.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    lang = request.form.get("language", "english")

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Save uploaded file
    filename  = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath  = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Step 1: YOLO — detect report regions
        detections = detect_regions(filepath)

        # Step 2: OCR — extract text
        # If YOLO found regions, OCR the most important one (findings)
        # Otherwise OCR the full image
        findings_region = next(
            (d for d in detections if "finding" in d["class"].lower()
             or "report" in d["class"].lower()), None
        )

        if findings_region:
            cropped_img = crop_region(filepath, findings_region["bbox"])
            extracted_text = extract_text_easyocr(cropped_img)
        else:
            extracted_text = extract_text_easyocr(Image.open(filepath))

        if not extracted_text.strip():
            extracted_text = "No readable text found in the report image."

        # Step 3: RAG — retrieve relevant medical context
        rag_context = retrieve(extracted_text[:300], top_k=2)

        # Step 4: LLM — generate explanation
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
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


# ── Route 2: Symptom checker ──────────────────────
@app.route("/api/check-symptoms", methods=["POST"])
def check_symptoms():
    """
    Accepts symptom text.
    Pipeline: BERT classification → RAG → LLM explanation
    """
    data = request.get_json()

    if not data or "symptoms" not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms = data.get("symptoms", "").strip()
    lang     = data.get("language", "english")

    if len(symptoms) < 10:
        return jsonify({"error": "Please describe your symptoms in more detail"}), 400

    try:
        # Step 1: BERT — classify symptoms
        bert_result = classify_symptoms(symptoms)

        # Step 2: RAG — retrieve relevant medical info
        rag_query   = f"{symptoms} {bert_result['category']}"
        rag_context = retrieve(rag_query, top_k=3)

        # Step 3: LLM — generate explanation
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