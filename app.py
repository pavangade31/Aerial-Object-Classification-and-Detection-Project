import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Page config
st.set_page_config(page_title="Bird vs Drone Classifier", layout="centered")
st.title("🕊️ Bird vs 🚁 Drone - Aerial Image Classifier")
st.write("Upload an aerial image for classification using MobileNetV2.")

@st.cache_resource
def load_model():
    # Rebuild the exact trained model architecture
    base_model = MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.load_weights("mobilenet_streamlit.weights.h5")

    return model

model = load_model()

uploaded_file = st.file_uploader("📂 Upload Image",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        with st.spinner("Processing..."):
            img = image.resize((160, 160))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0][0]

            label = "Drone" if prediction > 0.5 else "Bird"
            confidence = prediction if label == "Drone" else 1 - prediction

        st.success(f"Prediction: **{label}**")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
