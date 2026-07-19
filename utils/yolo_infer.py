from ultralytics import YOLO
from PIL import Image
import json
import os

MODEL_PATH = os.path.join("model", "yolo", "best.pt")

def detect_regions(image_path: str, conf_threshold: float = 0.4) -> list:
    """
    Run YOLOv8 on a medical report image.
    Returns list of detected regions with class name, confidence, and bounding box.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found at {MODEL_PATH}. "
            "Train on Colab first and download best.pt"
        )

    model = YOLO(MODEL_PATH)
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