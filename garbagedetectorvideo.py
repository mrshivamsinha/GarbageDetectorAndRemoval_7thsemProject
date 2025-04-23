import cv2
import torch
import math
import cvzone
from ultralytics import YOLO

# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Load the video
video_path = "Media/garbage6.mp4"  # Update with your video path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Perform object detection
    results = yolo_model(frame)

    # Loop through detections and draw green rectangles
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # If confidence is high, draw rectangle
            if conf > 0.3:
                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), colorC=(0, 255, 0), t=2)
                cvzone.putTextRect(frame, 'Garbage Detected', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(0, 255, 0))

    # Display the frame
    cv2.imshow("Garbage Detection - Video", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
