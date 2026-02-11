import re

def clean_text(ocr_data):
    all_texts = " ".join([item["text"] for item in ocr_data])

    cleaned = re.sub(r'[^A-Za-z0-9.,:₹%-]', " ", all_texts)
    return cleaned.lower()
