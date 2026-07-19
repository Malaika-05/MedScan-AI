# 🩺 MedScan AI — Multimodal Medical Report Explainer & Symptom Checker

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?style=for-the-badge&logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-mAP50%3A0.937-green?style=for-the-badge)
![BioBERT](https://img.shields.io/badge/BioBERT-94.8%25%20Accuracy-purple?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered multimodal system that helps patients in rural Pakistan understand medical reports and check symptoms — in English and Urdu.**

</div>

---

## 📌 The Problem

Pakistan has **33 million diabetics**, **600,000 new TB cases annually**, and millions of rural patients who travel 40+ km to see a doctor for a basic symptom check. When they do get a medical report, it is written in clinical English they cannot understand.

**MedScan AI bridges this gap** — a patient uploads their report or describes symptoms in plain language (English or Urdu), and gets a clear, friendly explanation with home remedies, consequences of ignoring the condition, and red flags that require immediate hospital care.

---

## ✨ Features

### 🔬 Multimodal Medical Report Analysis

- Upload a medical report image (JPG, PNG, PDF)
- **YOLOv8** detects and localizes report regions: `lab_name`, `medical_report`, `patient_personal_details`, `patient_report_details`, `report_authorization`, `report_logo`
- **EasyOCR / Pytesseract** extracts text from detected regions
- **Groq LLaMA 3.3 70B** explains findings in plain patient-friendly language

### 💬 AI Symptom Checker

- Patient describes symptoms in natural language
- **BioBERT** (fine-tuned on medical data) classifies across **24 disease categories**
- **RAG pipeline** retrieves relevant medical knowledge from a curated knowledge base
- **Groq LLM** generates a warm, doctor-friend style response

### 🌐 Bilingual Support — English & Urdu

- Toggle between English and Urdu responses
- Designed for patients in KPK and rural Pakistan who are more comfortable in Urdu

### 🤖 Friendly Doctor Personality

Every response follows a structured format:

- 🔍 What's happening — plain-language diagnosis
- 💊 Quick relief at home — practical Pakistani home remedies
- ⚠️ If you ignore this — honest consequences
- 🚨 Go to hospital immediately if — clear red flags
- 🏥 See a doctor — specific facility type + free government options

### 📚 Pakistan-Specific Knowledge Base

Curated medical knowledge covering diseases most prevalent in Pakistan:

- Malaria, Typhoid, TB, Dengue, Diabetes, Hypertension
- Gastroenteritis, Respiratory infections, Skin conditions, Mental health

---

## 🏗️ System Architecture

```
Patient Input (Image OR Text)
         │
         ▼
┌─────────────────────────────────────────────┐
│              Flask API Backend               │
│                                             │
│  Image Path              Text Path          │
│      │                      │               │
│      ▼                      ▼               │
│  YOLOv8              BioBERT Fine-tuned     │
│  Region Detection    Symptom Classifier     │
│  mAP50: 0.937        Accuracy: 94.8%        │
│      │                      │               │
│      ▼                      ▼               │
│   EasyOCR            RAG Pipeline           │
│  Text Extraction     FAISS + Sentence       │
│                      Transformers           │
│      │                      │               │
│      └──────────┬───────────┘               │
│                 ▼                           │
│         Groq LLaMA 3.3 70B                 │
│         Plain-language Response             │
└─────────────────────────────────────────────┘
         │
         ▼
  Patient-friendly explanation
  in English or Urdu
```

---

## 📊 Model Performance

| Model                            | Metric        | Score     |
| -------------------------------- | ------------- | --------- |
| YOLOv8n (Report Detection)       | mAP50         | **0.937** |
| YOLOv8n (Report Detection)       | mAP50-95      | **0.743** |
| YOLOv8n (Report Detection)       | Precision     | **0.895** |
| YOLOv8n (Report Detection)       | Recall        | **0.916** |
| BioBERT (Symptom Classification) | Test Accuracy | **94.8%** |
| BioBERT (Symptom Classification) | Macro F1      | **0.939** |
| BioBERT (Symptom Classification) | Weighted F1   | **0.939** |

**Disease Categories (24 classes):** Acne, Allergy, Arthritis, Bronchial Asthma, Cervical Spondylosis, Chicken Pox, Common Cold, Dengue, Diabetes, Dimorphic Hemorrhoids, Drug Reaction, Fungal Infection, GERD, Hypertension, Impetigo, Jaundice, Malaria, Migraine, Pneumonia, Psoriasis, Typhoid, Urinary Tract Infection, Varicose Veins

---

## 🛠️ Tech Stack

| Layer                | Technology                                         |
| -------------------- | -------------------------------------------------- |
| Computer Vision      | YOLOv8n (Ultralytics)                              |
| NLP / Classification | BioBERT (`dmis-lab/biobert-base-cased-v1.2`)       |
| OCR                  | EasyOCR + Pytesseract                              |
| RAG                  | FAISS + Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM                  | Groq API — LLaMA 3.3 70B Versatile                 |
| Backend              | Flask 3.0 + Flask-CORS                             |
| Frontend             | HTML5 + CSS3 + Vanilla JS                          |
| Deployment           | HuggingFace Spaces (Docker)                        |
| Training             | Google Colab (T4 GPU)                              |
| Labeling             | Roboflow                                           |

---

## 📁 Project Structure

```
MedScanAI/
├── app.py                          # Flask app entry point
├── Dockerfile                      # HuggingFace Spaces deployment
├── requirements.txt
├── .env                            # Local only — never commit
│
├── utils/
│   ├── yolo_infer.py               # YOLO region detection
│   ├── bert_infer.py               # BioBERT symptom classification
│   ├── ocr_utils.py                # EasyOCR + Tesseract text extraction
│   ├── rag_utils.py                # FAISS retrieval pipeline
│   └── llm_utils.py                # Groq LLM integration + prompts
│
├── app/
│   ├── templates/
│   │   └── index.html              # Frontend UI
│   └── static/
│
├── data/
│   ├── raw_reports/                # Sample medical report images
│   ├── labeled/                    # Roboflow labeled dataset
│   ├── symptoms/                   # Symptom2Disease dataset
│   ├── knowledge_base/             # Medical .txt files for RAG
│   │   ├── diabetes.txt
│   │   ├── malaria.txt
│   │   ├── tuberculosis.txt
│   │   ├── typhoid.txt
│   │   ├── hypertension.txt
│   │   ├── dengue.txt
│   │   ├── common_cold_flu.txt
│   │   ├── gastro_digestive.txt
│   │   ├── respiratory.txt
│   │   ├── skin_conditions.txt
│   │   └── mental_health.txt
│   └── uploads/                    # Temporary upload storage
│
├── models/
│   ├── yolo/
│   │   └── best.pt                 # Trained YOLOv8 weights
│   ├── bert/
│   │   ├── config.json
│   │   ├── model.safetensors       # Fine-tuned BioBERT weights
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── label_encoder.pkl
│   └── rag/
│       ├── faiss_index.bin         # FAISS vector index
│       └── chunks.pkl              # Knowledge base chunks
│
└── notebooks/
    ├── yolo_training.ipynb         # YOLOv8 Colab training notebook
    └── biobert_finetuning.ipynb    # BioBERT Colab fine-tuning notebook
```

---

## 🚀 Local Setup

### Prerequisites

- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Tesseract OCR binary ([Windows installer](https://github.com/UB-Mannheim/tesseract/wiki))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Malaika-05/MedScan-AI.git
cd MedScanAI

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
echo GROQ_API_KEY=your_key_here > .env

# 5. Build RAG index
python utils/rag_utils.py

# 6. Run the app
python app.py
```

Open `http://localhost:7860` in your browser.

> **Note:** `models/yolo/best.pt` and `models/bert/` are not included in the repository due to file size. Download them from the [HuggingFace Space](https://huggingface.co/spaces/YOUR_USERNAME/MedScanAI) or train your own using the notebooks in `notebooks/`.

---

## 🔌 API Endpoints

### `POST /api/check-symptoms`

Analyze symptom text and return AI-generated health advice.

**Request:**

```json
{
  "symptoms": "High fever for 3 days with chills and severe headache",
  "language": "english"
}
```

**Response:**

```json
{
  "success": true,
  "prediction": {
    "category": "Malaria",
    "confidence": 0.847,
    "top_3": [
      { "label": "Malaria", "score": 0.847 },
      { "label": "Dengue", "score": 0.091 },
      { "label": "Typhoid", "score": 0.062 }
    ]
  },
  "explanation": "🔍 What's happening: This looks like Malaria...",
  "rag_used": true
}
```

### `POST /api/upload-report`

Upload a medical report image for analysis.

**Request:** `multipart/form-data` with `file` and `language` fields.

**Response:**

```json
{
  "success": true,
  "detections": [
    {"class": "patient_report_details", "confidence": 0.962, "bbox": {...}},
    {"class": "lab_name", "confidence": 0.933, "bbox": {...}}
  ],
  "extracted_text": "Patient: ... Glucose: 180 mg/dL ...",
  "explanation": "🔍 What this report shows: ...",
  "regions_found": 2
}
```

---

## 🏋️ Training Your Own Models

### YOLOv8 — Medical Report Region Detection

1. Label medical report images on [Roboflow](https://roboflow.com)
2. Export in YOLOv8 format
3. Open `notebooks/yolo_training.ipynb` in Google Colab
4. Runtime → T4 GPU → Run all cells
5. Download `best.pt` → place in `models/yolo/`

### BioBERT — Symptom Classification

1. Download [Symptom2Disease dataset](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) from Kaggle
2. Open `notebooks/biobert_finetuning.ipynb` in Google Colab
3. Runtime → T4 GPU → Run all cells
4. Download `biobert_symptoms.zip` → extract to `models/bert/`

---

## 🌍 Social Impact

MedScan AI is built specifically for Pakistan's healthcare access gap:

- 🏥 **33 million** Pakistanis have diabetes — most undiagnosed
- 🦟 Pakistan is a **high-malaria burden country** — KPK and Sindh most affected
- 🦠 **600,000 new TB cases** annually — world's 5th highest burden
- 🌾 **Rural patients** travel 40+ km for basic consultations
- 📱 MedScan provides **free, instant, Urdu-language** AI health guidance

---

## ⚠️ Disclaimer

MedScan AI is an educational and informational tool only. It is **not a substitute for professional medical advice, diagnosis, or treatment.** Always consult a qualified doctor for medical decisions. In emergencies, call your nearest hospital immediately.

---

## 👩‍💻 Author

**Malaika Taqveem**

- 3rd Year AI Student — Abdul Wali Khan University Mardan (AWKUM)
- 📧 [LinkedIn](https://www.linkedin.com/in/malaika-taqveem-93354a285/)
- 🐙 [GitHub](https://github.com/Malaika-05)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with ❤️ for Pakistan's rural healthcare gap
</div>
