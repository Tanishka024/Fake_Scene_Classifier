import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import cv2
import tempfile
import os

from xai_occlusion import occlusion_xai
from ocr.ocr_engine import OCREngine
from ocr.postprocess import clean_text


# ------------------ PAGE CONFIG ------------------
try:
    icon = Image.open("logo.png")
except:
    icon = "🧠"

st.set_page_config(
    page_title="AI Fake Scene Classifier",
    page_icon=icon,
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #B5F0BD;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 AI Fake Scene Image Classifier")
st.write("Upload an image to check whether it is **Fake** or **Real**")


# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Tanishka024/fake-scene-classifier-model",
        filename="model.h5"
    )
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_ocr_engine():
    return OCREngine()


model = load_model()
ocr_engine = load_ocr_engine()


# ------------------ HELPERS ------------------
def save_uploaded_image(uploaded_file):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_path


# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ------------------ MAIN PIPELINE ------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------- CNN PREPROCESS --------
    image_resized = image.resize((224, 224))
    img_xai = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_xai, axis=0)

    pred = model.predict(img_array)[0][0]

    st.subheader("🧠 Prediction Result")

    if pred >= 0.68:
        predicted_label = "Real"
        predicted_conf = pred
        st.success(f"✅ Real Image\nConfidence: {predicted_conf:.2f}")
    else:
        predicted_label = "Fake"
        predicted_conf = 1 - pred
        st.error(f"❌ Fake Image\nConfidence: {predicted_conf:.2f}")

    # ------------------ OCR PIPELINE ------------------
    st.subheader("📄 OCR Analysis")

    run_ocr = st.checkbox("Run OCR on this image")

    if run_ocr:
        with st.spinner("Running OCR..."):
            img_path = save_uploaded_image(uploaded_file)
            ocr_output = ocr_engine.extract(img_path)
            cleaned_text = clean_text(ocr_output)

        if cleaned_text.strip():
            st.success("Text detected in image")

            st.text_area(
                "Extracted Text",
                cleaned_text,
                height=220
            )

            # -------- BASIC OCR HEURISTICS --------
            suspicious_words = [
                "copy", "duplicate", "sample", "fake",
                "edited", "photoshop"
            ]

            found_flags = [
                w for w in suspicious_words
                if w in cleaned_text.lower()
            ]

            if found_flags:
                st.error(
                    f"⚠️ Suspicious keywords detected: {', '.join(found_flags)}"
                )
            else:
                st.info("No obvious suspicious keywords detected")

        else:
            st.warning("No readable text detected in the image")

    # ------------------ XAI EXPLANATION ------------------
    explain = st.checkbox("🔍 Explain Prediction (Occlusion-based XAI)")

    if explain:
        st.subheader("📌 Model Explanation")

        regions, original_conf = occlusion_xai(
            model=model,
            image=img_xai,
            window_size=32,
            stride=16,
            top_k=5
        )

        explained_img = (img_xai * 255).astype(np.uint8)

        for r in regions:
            x, y = r["x"], r["y"]
            drop = r["drop"]

            cv2.rectangle(
                explained_img,
                (x, y),
                (x + 32, y + 32),
                (255, 0, 0),
                2
            )

            cv2.putText(
                explained_img,
                f"{drop:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )

        st.image(
            explained_img,
            caption="Top Influential Regions (Occlusion-based XAI)",
            use_column_width=True
        )

        max_drop = regions[0]["drop"] if regions else 0.0

        st.write(f"**Predicted Class:** {predicted_label}")
        st.write(f"**Prediction Confidence:** {predicted_conf:.2f}")
        st.write(f"**Maximum Confidence Drop:** {max_drop:.2f}")

        st.info(
            "Occlusion-based explanation highlights regions that caused "
            "the largest confidence drop when masked."
        )
