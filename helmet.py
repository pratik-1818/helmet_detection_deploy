import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Get the current directory where this script is running
current_dir = os.path.dirname(__file__)

# Set the model path using a relative path
model_path = os.path.join(current_dir, "best.pt")

# Debugging: Print model path
print(f"Loading YOLO model from: {model_path}")

# Google Drive Direct Download Link for `best.pt`
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?export=download&id=12XZTjBxaWkEwddcvaJMfLdsdXQsO4zoY"


# Get the current directory where this script is running
current_dir = os.path.dirname(__file__)

# Define model path
model_path = os.path.join(current_dir, "best.pt")

# Check if the model already exists, if not, download it
if not os.path.exists(model_path):
    print(f"Downloading best.pt from {GOOGLE_DRIVE_LINK}...")
    
    response = requests.get(GOOGLE_DRIVE_LINK)
    with open(model_path, "wb") as f:
        f.write(response.content)
    
    print("Download complete!")

# Load YOLOv8 model
model = YOLO("/home/vmukti/Desktop/helmet_detection/best.pt")

# Define class names
class_names = ["without helmet", "with helmet"]

def detect_helmet(frame):
    """
    Run YOLO detection on a frame.
    Returns the processed frame and detections.
    """
    results = model(frame)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if confidence > 0.6 and class_id < len(class_names):
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[class_id]
                color = (0, 255, 0) if class_name == "without helmet" else (0, 0, 255)
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                detections.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "box": [x1, y1, x2, y2]
                })
    
    return frame, detections
