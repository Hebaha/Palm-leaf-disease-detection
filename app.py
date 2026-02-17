import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input


# ----------------------------
# Load Model & Class Labels
# ----------------------------

MODEL_PATH = "EfficientNetB0_palm_disease_model.keras"
LABELS_PATH = "class_labels.json" 

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}
classes = [idx_to_class[i] for i in range(len(idx_to_class))]


# ----------------------------
# Severity Calculation
# ----------------------------

def softmax_entropy(p):
    p = np.clip(p, 1e-8, 1.0)
    return float(-np.sum(p * np.log(p)))

def severity_from_probs(probs, class_names):
    probs = np.asarray(probs, dtype=float)
    top = np.max(probs)
    ent = softmax_entropy(probs)
    top_cls = class_names[np.argmax(probs)]

    # Special case if class is healthy
    if top_cls.lower() == "healthy":
        return "Healthy"

    if ent > 1.20 or top < 0.55:
        return "Mild"
    if (0.55 <= top < 0.80) or (1.0 <= ent <= 1.20):
        return "Moderate"
    return "Severe"


# ----------------------------
# Preprocessing
# ----------------------------

def preprocess_image(img):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)       # VERY IMPORTANT FIX
    return arr


# ----------------------------
# Prediction Function
# ----------------------------

def predict(img):
    arr = preprocess_image(img)
    probs = model.predict(arr)[0]

    predicted_idx = np.argmax(probs)
    predicted_class = classes[predicted_idx]
    confidence = float(np.max(probs))
    severity = severity_from_probs(probs, classes)

    # Prepare top3 results like Gradio
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = {classes[i]: float(probs[i]) for i in top3_idx}

    return predicted_class, severity, confidence, top3


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Palm Leaf Disease Detection", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#22c55e;'>🌴 Palm Leaf Disease & Severity Detection</h1>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("---")

    with st.spinner("Analyzing Image..."):
        disease, severity, confidence, top3 = predict(img)

    st.subheader("📌 Prediction Results")
    st.write(f"**Disease:** {disease}")
    st.write(f"**Severity:** {severity}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    st.subheader("Top 3 Predictions")
    st.json(top3)

    # Severity alerts
    if severity == "Severe":
        st.error("⚠️ Severe Disease — Immediate action required!")
    elif severity == "Moderate":
        st.warning("⚠️ Moderate Disease — Needs attention.")
    elif severity == "Mild":
        st.info("ℹ️ Mild Infection — Monitor regularly.")
    else:
        st.success("✔ Healthy Palm Leaf")


st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with Streamlit + EfficientNetB0")
