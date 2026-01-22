import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

st.title("🧠 AI Fake Scene Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((224,224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success(f"🟢 REAL IMAGE ({prediction[0][0]*100:.2f}% confidence)")
    else:
        st.error(f"🔴 FAKE IMAGE ({(1-prediction[0][0])*100:.2f}% confidence)")
