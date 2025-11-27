import cv2
import streamlit as st
import supervision as sv
from inference import get_model
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Roboflow Live Inference", layout="wide")

st.title("🤖 Roboflow Webcam Inference")
st.markdown("Detected objects will be highlighted below with bounding boxes.")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Configuration")
# Replace with your defaults or keep blank to force manual entry
api_key = st.sidebar.text_input("cD8O59BRprZIhIp4jRxk")
model_id = st.sidebar.text_input("Model ID", value="cubik-cv-zyzo7/5")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# --- START/STOP BUTTONS ---
col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("Start Camera")
stop_btn = col2.button("Stop Camera")

# --- MAIN LOGIC ---
if start_btn and api_key and model_id:
    # 1. Load the Roboflow model
    try:
        model = get_model(model_id=model_id, api_key=api_key)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # 2. Initialize Annotators (Supervision)
    # BoxAnnotator draws the bounding boxes
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    # LabelAnnotator draws the class names and confidence
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1
    )

    # 3. Open Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam.")
        st.stop()

    # Create a placeholder for the video frame in the UI
    frame_placeholder = st.empty()
    stop_pressed = False

    while cap.isOpened() and not stop_pressed:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to fetch frame")
            break

        # 4. Run Inference
        # standard webcam frames are usually BGR, inference expects standard arrays
        results = model.infer(frame, confidence=confidence)[0]

        # 5. Process Detections
        detections = sv.Detections.from_inference(results)

        # 6. Annotate Frame
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )

        # 7. Display in Streamlit
        # Convert BGR (OpenCV) to RGB (Streamlit/Browser)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Check if stop button is clicked in the UI sidebar
        # (Note: Streamlit buttons reset on re-run, so we use the stop_btn logic outside loop in a real app,
        # but for a simple loop, we rely on the user refreshing or hitting 'Stop Camera' which triggers a rerun)
        # To make "Stop" work smoothly in this loop structure without complex session state:
        # We rely on the user toggling the script or closing the tab for "Simple" usage.
    
    cap.release()

elif start_btn and not api_key:
    st.error("Please enter your Roboflow API Key in the sidebar.")
