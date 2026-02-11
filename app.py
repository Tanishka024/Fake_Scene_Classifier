import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
import os
import sys

from ocr_engine_simple import extract_text
from field_extractor import extract_fields
from risk_engine import calculate_risk_score
from xai_occlusion import occlusion_xai

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

try:
    icon = Image.open("logo.png")
except Exception:
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

st.markdown("<h1>🔍 AI Fake Scene Image Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:1.1rem'>Upload an image to verify authenticity using AI-powered forensics</div>",
    unsafe_allow_html=True
)

mode = st.radio(
    "Select Analysis Type",
    [
        "📷 Normal Image (Photos / Scenes)",
        "📄 Text-based Image (Payments / Documents)"
    ]
)

@st.cache_resource(show_spinner=True)
def load_model():
    model_path = hf_hub_download(
        repo_id="Tanishka024/fake-scene-classifier-model",
        filename="model.h5"
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

if mode == "📷 Normal Image (Photos / Scenes)":

    uploaded_file = st.file_uploader(
        "Upload a photo image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_resized = image.resize((224, 224))
        img_xai = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_xai, axis=0)

        pred = float(model.predict(img_array, verbose=0)[0][0])

        extracted_text = extract_text(image)
        text_lower = extracted_text.lower()

        penalty = 0

        suspicious_words = [
            "payment successful",
            "upi",
            "transaction id",
            "thanks app",
            "prepaid",
            "bank reference",
            "posting id"
        ]

        for w in suspicious_words:
            if w in text_lower:
                penalty += 0.08

        gray = np.array(image.convert("L"))
        if np.std(gray) < 10:
            penalty += 0.25

        final_score = pred - penalty
        final_score = max(0, min(final_score, 1))

        st.subheader("🧠 Prediction Result")

        if final_score >= 0.70:
            predicted_label = "Real"
            predicted_conf = final_score
            st.success(f"✅ Real Image\nConfidence: {predicted_conf:.2f}")
        else:
            predicted_label = "Fake"
            predicted_conf = 1 - final_score
            st.error(f"❌ Fake Image\nConfidence: {predicted_conf:.2f}")


        explain = st.checkbox("🔍 Explain Prediction (Occlusion-based XAI)")

        if explain:
            regions, _ = occlusion_xai(
                model=model,
                image=img_xai,
                window_size=32,
                stride=16,
                top_k=5
            )

            explained_img = (img_xai * 255).astype(np.uint8)
            pil_img = Image.fromarray(explained_img)
            draw = ImageDraw.Draw(pil_img)

            if regions:
                for r in regions:
                    x, y = r["x"], r["y"]
                    drop = r["drop"]

                    draw.rectangle(
                        [(x, y), (x + 32, y + 32)],
                        outline=(255, 0, 0),
                        width=2
                    )

                    draw.text(
                        (x, max(y - 10, 5)),
                        f"{drop:.2f}",
                        fill=(255, 0, 0)
                    )

                max_drop = regions[0]["drop"]
            else:
                max_drop = 0.0

            st.image(
                np.array(pil_img),
                caption="Top Influential Regions",
                use_column_width=True
            )

            st.write(f"**Predicted Class:** {predicted_label}")
            st.write(f"**Prediction Confidence:** {predicted_conf:.2f}")
            st.write(f"**Maximum Confidence Drop:** {max_drop:.2f}")

elif mode == "📄 Text-based Image (Payments / Documents)":

    st.markdown("### 📄 Document & Payment Screenshot Analysis")

    uploaded_doc = st.file_uploader(
        "Upload a payment screenshot or document image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_doc is not None:
        doc_image = Image.open(uploaded_doc).convert("RGB")

        st.image(
            doc_image,
            caption="Uploaded Document / Screenshot",
            use_column_width=True
        )

        if st.button("🔍 Run OCR Analysis"):
            extracted_text = extract_text(doc_image)

            if not extracted_text.strip():
                st.warning("No readable text detected.")
            else:
                st.text_area(
                    "📝 Extracted Text",
                    extracted_text,
                    height=260
                )

                fields = extract_fields(extracted_text)

                st.subheader("📌 Extracted Fields")
                st.json(fields)

                risk_result = calculate_risk_score(fields, extracted_text)

                st.subheader("🚨 Risk Assessment")

                if risk_result["risk_level"] == "Low Risk":
                    st.success(f"✅ {risk_result['risk_level']} (Score: {risk_result['risk_score']})")
                elif risk_result["risk_level"] == "Medium Risk":
                    st.warning(f"⚠️ {risk_result['risk_level']} (Score: {risk_result['risk_score']})")
                else:
                    st.error(f"❌ {risk_result['risk_level']} (Score: {risk_result['risk_score']})")

                for reason in risk_result["reasons"]:
                    st.write(f"- {reason}")