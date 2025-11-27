import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from inference_sdk import InferenceHTTPClient
import cv2
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import av
import threading  # For potential thread safety if needed

# Set page config
st.set_page_config(
    page_title="Cathay Pacific AI Food Waste Detector",
    layout="wide",
    page_icon="✈️"
)

# Green theme CSS inspired by Cathay Pacific colors (primary green: #00645A, light background: #f0f8f5)
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #f0f8f5;
    }
    [data-testid="stSidebar"] {
        background-color: #00645A;
        color: white;
    }
    h1, h2, h3 {
        color: #00645A;
    }
    .stButton > button {
        background-color: #00645A;
        color: white;
    }
    .stSlider .stSliderLabel {
        color: #00645A;
    }
</style>
""", unsafe_allow_html=True)

# Placeholder for model details - replace with your actual Roboflow model ID and API key
MODEL_ID = "cubik-cv-zyzo7/7"  # Replace with your model ID, e.g., "airline-food-waste-detection/1"
API_KEY = "cD8O59BRprZIhIp4jRxk"  # Replace with your Roboflow API key (ensure it's valid to avoid 403 errors)

# Initialize Inference Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY)

# Class names (assuming your model is trained on these 14 classes)
CLASS_NAMES = [
    "yogurt_ice_cream_eaten", "yogurt_ice_cream_untouched",
    "salad_eaten", "salad_untouched",
    "main_eaten", "main_untouched",
    "jam_eaten", "jam_untouched",
    "butter_eaten", "butter_untouched",
    "bread_eaten", "bread_untouched",
    "beverage_drunk", "beverage_untouched"
]

# Categories for analytics
CATEGORIES = ["yogurt_ice_cream", "salad", "main", "jam", "butter", "bread", "beverage"]

# Main title
st.title("Cathay Pacific AI Food Waste Detector")
st.markdown("Use your webcam to scan airline food trays. The app detects items and provides real-time waste analytics.")

# Initialize Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

# Define VideoProcessor class to handle frame processing and store detections
class VideoProcessor:
    def __init__(self):
        self.detections = None
        self._lock = threading.Lock()  # For thread safety when accessing detections

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Convert frame to numpy array (OpenCV format)
        image = frame.to_ndarray(format="bgr24")

        # 2. Run Inference
        try:
            results = client.infer(image, model_id=MODEL_ID)
            
            # 3. Process Results
            detections = sv.Detections.from_inference(results)

            # Store detections with lock for thread safety
            with self._lock:
                self.detections = detections

            # 4. Annotate Frame
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            
        except Exception as e:
            # If API fails (e.g., rate limit or 403), just return the original frame
            print(f"Error: {e}")
            annotated_frame = image

        # 5. Return the annotated frame back to the browser
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- WEBRTC STREAMER ---
# Use video_processor_factory to create an instance of VideoProcessor
ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Analytics section
if ctx.state.playing and ctx.video_processor:
    # Access detections with lock for safety
    with ctx.video_processor._lock:
        detections = ctx.video_processor.detections
    
    if detections is not None and len(detections) > 0:
        st.subheader("Detected Items and Waste Analytics")

        # Create DataFrame from detections
        class_ids = detections.class_id
        confidences = detections.confidence
        class_names = [CLASS_NAMES[id] for id in class_ids]

        df = pd.DataFrame({
            "Item": class_names,
            "Confidence": confidences
        })

        # Display detected items table
        st.table(df)

        # Calculate waste per category
        waste_data = {}
        overall_eaten = 0
        overall_untouched = 0

        for cat in CATEGORIES:
            eaten_count = sum(1 for name in class_names if name == f"{cat}_eaten")
            untouched_count = sum(1 for name in class_names if name == f"{cat}_untouched")
            total = eaten_count + untouched_count
            waste_perc = (untouched_count / total * 100) if total > 0 else 0
            waste_data[cat] = {
                "Eaten": eaten_count,
                "Untouched": untouched_count,
                "Waste %": waste_perc
            }
            overall_eaten += eaten_count
            overall_untouched += untouched_count

        # Waste DataFrame
        waste_df = pd.DataFrame.from_dict(waste_data, orient="index")
        st.table(waste_df)

        # Overall waste percentage
        overall_total = overall_eaten + overall_untouched
        overall_waste_perc = (overall_untouched / overall_total * 100) if overall_total > 0 else 0
        st.metric("Overall Waste Percentage", f"{overall_waste_perc:.2f}%")

        # Bar chart for waste %
        fig, ax = plt.subplots()
        waste_df["Waste %"].plot(kind="bar", ax=ax, color="#00645A")
        ax.set_ylabel("Waste Percentage")
        ax.set_title("Waste Percentage by Category")
        st.pyplot(fig)

        # Pie chart for overall
        if overall_total > 0:
            pie_fig, pie_ax = plt.subplots()
            pie_ax.pie([overall_eaten, overall_untouched], labels=["Eaten", "Untouched"], autopct="%1.1f%%", colors=["#00645A", "#E70013"])
            pie_ax.set_title("Overall Food Status")
            st.pyplot(pie_fig)
    else:
        st.info("No detections yet. Point the camera at a food tray.")

# Footer
st.markdown("---")
st.markdown("Made by Wilson C. @CUBIK")
st.markdown("Powered by Streamlit, Roboflow Inference, and Supervision.")
st.markdown("Designed for Cathay Pacific sustainability initiatives.")
