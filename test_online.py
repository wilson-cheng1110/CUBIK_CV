import cv2
import streamlit as st
import supervision as sv
from inference import get_model
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# --- PAGE CONFIG ---
st.set_page_config(page_title="Roboflow Live Inference", layout="wide")

st.title("🤖 Roboflow Webcam Inference")
st.markdown("Detected objects will be highlighted in real-time from your browser's webcam.")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Configuration")
api_key = "cD8O59BRprZIhIp4jRxk"
model_id = "cubik-cv-zyzo7/7"
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Use session state to track if streaming is active
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

# --- START/STOP BUTTONS ---
col1, col2 = st.sidebar.columns(2)
if col1.button("Start Inference"):
    if api_key and model_id:
        st.session_state.streaming = True
    else:
        st.error("Please provide API Key and Model ID.")
if col2.button("Stop Inference"):
    st.session_state.streaming = False

# --- MAIN LOGIC ---
if st.session_state.streaming:
    # Load model once (outside the transformer to avoid reloading per frame)
    try:
        model = get_model(model_id=model_id, api_key=api_key)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state.streaming = False
        st.stop()

    # Initialize Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    class RoboflowTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV-compatible array

            # Run Inference
            results = model.infer(img, confidence=confidence)[0]

            # Process Detections
            detections = sv.Detections.from_inference(results)

            # Annotate Frame
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            return annotated_frame  # Return annotated NumPy array (auto-converted to frame)

    # Start WebRTC streamer (handles client-side camera)
    webrtc_streamer(
        key="roboflow-inference",
        mode=WebRtcMode.SENDRECV,  # Send video to server and receive back
        video_transformer_factory=RoboflowTransformer,
        async_transform=True,  # Process frames asynchronously for better performance
    )