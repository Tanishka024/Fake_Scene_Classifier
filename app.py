import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


st.set_page_config(
    page_title="AI Fake Scene Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Fake Scene Image Classifier")
st.write("Upload an image to check whether it is **Fake** or **Real**")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Tanishka024/fake-scene-classifier-model",
        filename="model.h5"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)


    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")

    if prediction >= 0.5:
        st.success(f"✅ Real Image  \nConfidence: {prediction:.2f}")
    else:
        st.error(f"❌ Fake Image  \nConfidence: {1 - prediction:.2f}")
