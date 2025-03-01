import cv2
import math
import cvzone
import os
from ultralytics import YOLO

# Load YOLO model with custom weights
model_path = "Weights/best.pt"
image_path = "Media/image1.png"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at {image_path}")

yolo_model = YOLO(model_path)

# Define class names
class_labels = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

# Load the image
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Error loading image. Check the path and format.")

# Perform object detection
results = yolo_model(img)

# Loop through the detections and draw bounding boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.3 and cls < len(class_labels):
            # Draw a small box around the detected object
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green rectangle

            # Write "Garbage Detected" on the box
            cv2.putText(img, "Garbage Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the image with detections
cv2.imshow("Detected Image", img)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()