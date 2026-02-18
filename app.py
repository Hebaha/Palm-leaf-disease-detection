import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. إعدادات الصفحة والمسارات ---
st.set_page_config(page_title="Nekhlawi - Palm Disease Detection", page_icon="🌴")

# تأكدي أن هذه الملفات مرفوعة بنفس الأسماء في GitHub
MODEL_PATH = "my_palm_model.h5" 
LABELS_PATH = "class_labels.json"

# --- 2. دالة تحميل الموديل بوضع التوافق ---
@st.cache_resource
def load_palm_model():
    try:
        # استخدام compile=False يتخطى أخطاء إصدارات Keras المختفلة
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. تحميل البيانات ---
model = load_palm_model()

try:
    with open(LABELS_PATH, "r") as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
except Exception as e:
    st.error(f"Error loading labels: {e}")
    classes = []

# --- 4. دالة المعالجة والتنبؤ ---
def predict(img):
    # تجهيز الصورة بحجم 224x224 كما في تدريب EfficientNetB0
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    # إجراء التنبؤ
    preds = model.predict(arr)
    
    # حل مشكلة المصفوفات المتداخلة (Tensors)
    if isinstance(preds, (list, tuple)):
        probs = preds[0]
    else:
        probs = preds
        
    if len(probs.shape) > 1:
        probs = probs[0]

    predicted_idx = np.argmax(probs)
    confidence = float(np.max(probs))
    
    return classes[predicted_idx], confidence

# --- 5. واجهة المستخدم (UI) ---
st.markdown("<h1 style='text-align:center; color:#22c55e;'>🌴 Nekhlawi: Palm Disease Detection</h1>", unsafe_allow_html=True)
st.write("Welcome, Hebah! Upload a palm leaf image to diagnose its health.")

uploaded_file = st.file_uploader("Choose a palm leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if model is not None and classes:
        with st.spinner('Analyzing the image...'):
            label, confidence = predict(image)
            
        # عرض النتيجة النهائية
        st.markdown(f"### Result: **{label}**")
        st.progress(confidence)
        st.write(f"Confidence Level: **{confidence*100:.2f}%**")
        
        # تنبيه بناءً على النتيجة (Severity Alert)
        if "Healthy" in label:
            st.success("The palm appears to be healthy!")
        else:
            st.warning("Action may be required. Please check the disease details.")
    else:
        st.error("Model or labels are not loaded correctly.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Nekhlawi Project - Built with Streamlit & EfficientNetB0")
