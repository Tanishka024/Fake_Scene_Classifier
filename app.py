from fastapi import FastAPI, UploadFile, File
import os, uuid, shutil
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from ocr.ocr_engine import OCREngine
from ocr.postprocess import clean_text

# -------------------------
# App & Model Load
# -------------------------
app = FastAPI()
model = load_model("model.h5")
ocr_engine = OCREngine()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# Image Preprocessing
# -------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 1️⃣ Save image
    image_id = str(uuid.uuid4())
    image_path = f"{UPLOAD_DIR}/{image_id}.jpg"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2️⃣ Fake image prediction
    img_input = preprocess_image(image_path)
    prediction = model.predict(img_input)[0][0]

    is_fake = bool(prediction > 0.5)

    # 3️⃣ OCR only if image likely contains text
    ocr_text = None
    if is_fake:   # (you can change this condition later)
        ocr_output = ocr_engine.extract(image_path)
        ocr_text = clean_text(ocr_output)

    # 4️⃣ Response
    return {
        "fake_probability": float(prediction),
        "is_fake": is_fake,
        "ocr_text": ocr_text
    }
