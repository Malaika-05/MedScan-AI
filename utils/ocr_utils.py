from PIL import Image
import pytesseract
import easyocr
import numpy as np
import os

# EasyOCR reader — loaded once
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        print("Loading EasyOCR (first load takes ~30 seconds)...")
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text_tesseract(image: Image.Image) -> str:
    """
    Extract text from a PIL Image using Tesseract OCR.
    Faster but less accurate than EasyOCR on messy scans.
    """
    text = pytesseract.image_to_string(image, lang="eng")
    return text.strip()


def extract_text_easyocr(image: Image.Image) -> str:
    """
    Extract text from a PIL Image using EasyOCR.
    Better for low-quality scans and mixed fonts.
    """
    reader = _get_reader()
    img_array = np.array(image)
    results = reader.readtext(img_array, detail=0, paragraph=True)
    return " ".join(results).strip()


def extract_text_from_path(image_path: str, method: str = "easyocr") -> str:
    """
    Extract text from an image file path.
    method: 'easyocr' (default) or 'tesseract'
    """
    img = Image.open(image_path).convert("RGB")

    if method == "tesseract":
        return extract_text_tesseract(img)
    else:
        return extract_text_easyocr(img)


if __name__ == "__main__":
    test_image = os.path.join("data", "raw_reports", "sample.jpg")

    if not os.path.exists(test_image):
        print("Place a test image at data/raw_reports/sample.jpg")
    else:
        print("Testing EasyOCR...")
        text = extract_text_from_path(test_image, method="easyocr")
        print("Extracted text:")
        print(text[:500])