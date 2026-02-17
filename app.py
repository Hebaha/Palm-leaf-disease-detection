import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. إعداد المسارات
MODEL_PATH = "EfficientNetB0_palm_disease_model.keras"
LABELS_PATH = "class_labels.json" 
# استخدام compile=False يتخطى العديد من تعارضات الإعدادات
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# 2. دالة تحميل الموديل بوضع التوافق
@st.cache_resource
def load_palm_model():
    # استخدام الطريقة التقليدية الأكثر استقراراً
    return model

try:
    model = load_palm_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# تحميل أسماء الأمراض
with open(LABELS_PATH, "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}
classes = [idx_to_class[i] for i in range(len(idx_to_class))]

# 3. دالة التنبؤ المحمية
def predict(img):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    preds = model.predict(arr)
    
    # حل مشكلة المدخلات المزدوجة (Layer expects 1 input but received 2)
    if isinstance(preds, (list, tuple)):
        probs = preds[0]
    else:
        probs = preds
        
    if len(probs.shape) > 1:
        probs = probs[0]

    predicted_idx = np.argmax(probs)
    return classes[predicted_idx], float(np.max(probs))

# --- الواجهة الرسومية ---
st.markdown("<h1 style='text-align:center; color:#22c55e;'>🌴 Nekhlawi: Disease Detection</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Palm Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    with st.spinner("جاري التحليل..."):
        label, conf = predict(image)
        
    st.success(f"النتيجة: {label}")
    st.info(f"نسبة الثقة: {conf*100:.2f}%")
