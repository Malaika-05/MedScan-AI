from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import os

MODEL_PATH = os.path.join("model", "bert")
LABEL_ENCODER_PATH = os.path.join("model", "bert", "label_encoder.pkl")

_model = None
_tokenizer = None
_label_encoder = None


def _load_model():
    """Load model once and cache it in memory."""
    global _model, _tokenizer, _label_encoder

    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"BERT model not found at {MODEL_PATH}. "
            "Fine-tune on Colab first and download the model folder."
        )

    print("Loading BioBERT model...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.eval()

    with open(LABEL_ENCODER_PATH, "rb") as f:
        _label_encoder = pickle.load(f)

    print("BioBERT loaded successfully")


def classify_symptoms(text: str) -> dict:
    """
    Classify symptom text using fine-tuned BioBERT.
    Returns predicted disease category, confidence score, and top 3 predictions.
    """
    _load_model()

    encoding = _tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = _model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = probs.argmax().item()
    confidence = probs.max().item()

    top3_indices = probs.topk(3).indices.tolist()
    top3 = [
        {
            "label": _label_encoder.classes_[i],
            "score": round(probs[i].item(), 3)
        }
        for i in top3_indices
    ]

    return {
        "category": _label_encoder.inverse_transform([pred_id])[0],
        "confidence": round(confidence, 3),
        "top_3": top3
    }


if __name__ == "__main__":
    test_inputs = [
        "Patient has high fever 103F, severe headache, stiff neck, and sensitivity to light",
        "Chest pain radiating to left arm, shortness of breath, sweating",
        "Persistent cough for 3 weeks, blood in sputum, night sweats, weight loss",
    ]

    print("Testing BioBERT symptom classifier...")
    print("(Model will show FileNotFoundError until you download from Colab)\n")

    for text in test_inputs:
        try:
            result = classify_symptoms(text)
            print(f"Input: {text[:60]}...")
            print(f"Predicted: {result['category']} ({result['confidence']*100:.1f}%)")
            print(f"Top 3: {[x['label'] for x in result['top_3']]}\n")
        except FileNotFoundError as e:
            print(f"Expected error (model not trained yet): {e}")
            break