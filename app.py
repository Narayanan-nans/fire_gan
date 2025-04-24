import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

st.set_page_config(page_title="🔥 Fire Detection", layout="wide")

# Load the TFLite model once
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="forest_fire_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img_height, img_width = input_details[0]['shape'][1:3]

# Detect fire
def detect_fire(frame):
    resized = cv2.resize(frame, (img_width, img_height))
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return prediction > 0.01

# Initialize Streamlit state
if "running" not in st.session_state:
    st.session_state.running = False

st.markdown("<h1 style='text-align: center;'>🔥 Real-Time Fire Detection</h1>", unsafe_allow_html=True)

status_col, button_col = st.columns([1, 1])

with button_col:
    st.markdown("### 🎮 Controls")
    if st.button("▶️ Start Detection"):
        st.session_state.running = True
    if st.button("⏹️ Stop Detection"):
        st.session_state.running = False

with status_col:
    status = "🔴 Live" if st.session_state.running else "⚪️ Stopped"
    st.markdown(f"### 🟢 Status: {status}")

frame_placeholder = st.empty()
info_placeholder = st.empty()

# Capture from webcam
cap = cv2.VideoCapture(0)

# Streamlit loop
while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        info_placeholder.warning("⚠️ Webcam feed failed.")
        break

    fire_detected = detect_fire(frame)
    label = "🔥 Fire Detected!" if fire_detected else "✅ No Fire"
    color = (0, 0, 255) if fire_detected else (0, 255, 0)

    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)

    frame_placeholder.image(frame_image, caption=label, use_container_width=True)
    info_placeholder.info(f"🧠 Detection Result: **{label}**")

    # Check again if running (user might have stopped)
    if not st.session_state.running:
        break

cap.release()
