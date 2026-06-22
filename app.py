import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.set_page_config(
    page_title="Grooming Compliance Checker",
    page_icon="🛡️"
)

st.title("🛡️ Grooming Compliance Checker")

st.write(
    "Upload an employee image to verify grooming compliance."
)

# -----------------------------
# Google Drive Model Settings
# -----------------------------
FILE_ID = "1bYvI2nTfBIHM8zrhtMPHeYuoDW8lu8RL"

MODEL_NAME = "grooming_model.h5"

MODEL_PATH = os.path.join(
    os.getcwd(),
    MODEL_NAME
)

DOWNLOAD_URL = (
    f"https://drive.google.com/uc?id={FILE_ID}"
)

# -----------------------------
# Load Model (Download if needed)
# -----------------------------
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.write(
            "Downloading model from Google Drive..."
        )

        gdown.download(
            DOWNLOAD_URL,
            MODEL_PATH,
            quiet=False
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()

st.success("✅ Model loaded successfully")

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Employee Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(
        img,
        axis=0
    )

    # Prediction
    prob = model.predict(
        img,
        verbose=0
    )[0][0]

    reject_score = prob * 100
    accept_score = 100 - reject_score

    st.subheader(
        "Analysis Result"
    )

    if prob < 0.5:

        st.success(
            "✅ ACCEPTED"
        )

        st.metric(
            "Compliance Score",
            f"{accept_score:.2f}%"
        )

        st.info(
            "Hair and facial grooming standards satisfied."
        )

    else:

        st.error(
            "❌ REJECTED"
        )

        st.metric(
            "Non-Compliance Score",
            f"{reject_score:.2f}%"
        )

        st.warning(
            "Rejected: Facial grooming exceeds permissible threshold."
        )

