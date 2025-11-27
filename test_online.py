import streamlit as st
import av
import supervision as sv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Roboflow Live Inference", layout="centered")

st.title("☁ Public Host Object Detection")
st.write("This app streams video from YOUR client browser to the server for processing.")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Settings")
# We use st.secrets for safety on public hosts, but allow manual entry for testing
# On Streamlit Cloud, set these in the "Secrets" management tab.
ROBOFLOW_API_KEY = st.sidebar.text_input("Roboflow API Key", type="password")
MODEL_ID = st.sidebar.text_input("Model ID", value="yolov8n-640")

# --- MAIN LOGIC ---

if not ROBOFLOW_API_KEY:
    st.error("Please provide a Roboflow API Key to proceed.")
    st.stop()

# Initialize the Roboflow Client
# We use InferenceHTTPClient here because it is lighter on the server's memory
# than loading the full model weights locally on a free-tier cloud instance.
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# Initialize Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

# Definition of the callback function
# This function runs inside a separate thread for every single video frame
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    # 1. Convert frame to numpy array (OpenCV format)
    image = frame.to_ndarray(format="bgr24")

    # 2. Run Inference
    # We send the image to Roboflow's API. 
    # Note: For high FPS, a local model is better, but this is safest for public cloud memory.
    try:
        results = client.infer(image, model_id=MODEL_ID)
        
        # 3. Process Results
        # The HTTP client returns a dictionary, we convert it to supervision Detections
        detections = sv.Detections.from_inference(results)

        # 4. Annotate Frame
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
    except Exception as e:
        # If API fails (e.g., rate limit), just return the original frame
        print(f"Error: {e}")
        annotated_frame = image

    # 5. Return the annotated frame back to the browser
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- WEBRTC STREAMER ---
# This is the magic component that replaces cv2.VideoCapture
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
