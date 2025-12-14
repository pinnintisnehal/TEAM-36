import easyocr
import re
import numpy as np
from PIL import Image
from difflib import SequenceMatcher

reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_image(image):
    img = Image.open(image).convert("RGB")
    img_np = np.array(img)
    result = reader.readtext(img_np, detail=0)
    return " ".join(result).lower()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def identify_drug_name(image, available_products, threshold=0.6):
    raw_text = extract_text_from_image(image)
    text = clean_text(raw_text)

    best_match = None
    best_score = 0

    for product in available_products:
        product_lower = product.lower()
        if product_lower in text:
            return product, text


    if best_score >= threshold:
        return best_match, text

    return None, text
