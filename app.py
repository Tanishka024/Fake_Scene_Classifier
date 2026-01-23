import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import cv2
from xai_occlusion import occlusion_xai

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Fake Scene Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Fake Scene Image Classifier")
st.write("Upload an image to check whether it is **Fake** or **Real**")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Tanishka024/fake-scene-classifier-model",
        filename="model.h5"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- Main Logic ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------- Preprocessing ----------
    image_resized = image.resize((224, 224))
    img_xai = np.array(image_resized) / 255.0      # (224,224,3)
    img_array = np.expand_dims(img_xai, axis=0)    # (1,224,224,3)

    # ---------- Prediction ----------
    pred = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")

    if pred >= 0.5:
        predicted_label = "Real"
        predicted_conf = pred
        st.success(f"✅ Real Image\nConfidence: {predicted_conf:.2f}")
    else:
        predicted_label = "Fake"
        predicted_conf = 1 - pred
        st.error(f"❌ Fake Image\nConfidence: {predicted_conf:.2f}")

    # ---------- XAI Toggle ----------
    explain = st.checkbox("🔍 Explain Prediction (Occlusion-based XAI)")

    if explain:
        st.subheader("Model Explanation")

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

        max_drop = regions[0]["drop"] if len(regions) > 0 else 0.0

        st.write(f"**Predicted Class:** {predicted_label}")
        st.write(f"**Prediction Confidence:** {predicted_conf:.2f}")
        st.write(f"**Maximum Confidence Drop:** {max_drop:.2f}")

        st.info(
            "This explanation is generated using occlusion-based sliding window analysis. "
            "Regions that caused the largest confidence drop are highlighted, indicating "
            "strong influence on the model’s decision."
        )