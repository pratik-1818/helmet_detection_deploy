from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import os
import requests
from helmet import detect_helmet  # Import YOLO detection function

app = FastAPI()

# ✅ Corrected Google Drive Direct Download Link for Video
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?export=download&id=1OditoY-FRxHBJ9E8XUKqSn6e84VM8WU3"

# Define video path
video_path = "video.mp4"

# Check if the video file exists, if not, download it
if not os.path.exists(video_path) or os.path.getsize(video_path) < 50000:  # Ensure file is valid
    print(f"Downloading video from {GOOGLE_DRIVE_LINK}...")

    response = requests.get(GOOGLE_DRIVE_LINK, stream=True)
    with open(video_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    print(f"Download complete! File size: {os.path.getsize(video_path)} bytes")

def generate_frames():
    cap = cv2.VideoCapture(video_path)  # Load video

    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return None  # Stop if video cannot be opened

    print("✅ Video file opened successfully!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ End of video reached!")
            break

        # Run YOLO detection
        frame, detections = detect_helmet(frame)

        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_bytes = encoded_frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/")
def home():
    return {"message": "Helmet Detection API is Running with Video!"}

@app.get("/video_feed/")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
