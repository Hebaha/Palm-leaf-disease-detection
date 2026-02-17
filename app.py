import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. تحميل الموديل والملصقات بأمان
MODEL_PATH = "EfficientNetB0_palm_disease_model.keras"
LABELS_PATH = "class_labels.json"

# ----------------------------
# Load Model & Class Labels
# ----------------------------

MODEL_PATH = "EfficientNetB0_palm_disease_model.keras"
LABELS_PATH = "class_labels.json" 

@st.cache_resource
def load_palm_model():
    # استخدام التنسيق الحديث لـ Keras 3 وتخطي إعادة البناء
    return tf.keras.saving.load_model(MODEL_PATH, compile=False, safe_mode=False)

try:
    model = load_palm_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

with open(LABELS_PATH, "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}
classes = [idx_to_class[i] for i in range(len(idx_to_class))]

# 2. دالة المعالجة المسبقة
def preprocess_image(img):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

# 3. دالة التنبؤ (مع معالجة خطأ الـ Tensor المزدوج)
def predict(img):
    arr = preprocess_image(img)
    preds = model.predict(arr)
    
    # حل مشكلة الخطأ: إذا أرجع الموديل قائمة، نأخذ العنصر الأول فقط
    if isinstance(preds, list):
        probs = preds[0][0]
    else:
        probs = preds[0]

    predicted_idx = np.argmax(probs)
    return classes[predicted_idx], float(np.max(probs)), probs

# --- واجهة المستخدم (UI) كما هي في مشروعك ---
st.title("🌴 Nekhlawi: Palm Disease Detection")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, use_column_width=True)
    
    label, confidence, all_probs = predict(img)
    st.success(f"المرض المكتشف: {label}")
    st.info(f"نسبة الثقة: {confidence*100:.2f}%")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with Streamlit + EfficientNetB0")
