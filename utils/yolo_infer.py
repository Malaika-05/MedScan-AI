from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download
import json
import os

_model = None  # cached after first load

def get_model():
    global _model
    if _model is None:
        weights_path = hf_hub_download(
            repo_id="openmodels05/medscan-yolo",
            filename="best.pt"
        )
        _model = YOLO(weights_path)
    return _model

def detect_regions(image_path: str, conf_threshold: float = 0.4) -> list:
    """
    Run YOLOv8 on a medical report image.
    Returns list of detected regions with class name, confidence, and bounding box.
    """
    model = get_model()
    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 1),
                    "y1": round(float(box.xyxy[0][1]), 1),
                    "x2": round(float(box.xyxy[0][2]), 1),
                    "y2": round(float(box.xyxy[0][3]), 1),
                }
            })
    return detections


def crop_region(image_path: str, bbox: dict) -> Image.Image:
    """
    Crop a detected region from the original image.
    Used to pass cropped region to OCR.
    """
    img = Image.open(image_path).convert("RGB")
    cropped = img.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
    return cropped


if __name__ == "__main__":
    # Quick test — replace with your actual image path
    test_image = os.path.join("data", "raw_reports", "sample.jpg")

    if not os.path.exists(test_image):
        print(f"Test image not found at {test_image}")
        print("Place a medical report image at data/raw_reports/sample.jpg")
    else:
        print("Running YOLO detection...")
        result = detect_regions(test_image)
        print(json.dumps(result, indent=2))
        print(f"\nDetected {len(result)} regions")