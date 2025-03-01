import cv2
import math
import cvzone
from ultralytics import YOLO

# Initialize video capture
video_path = "Media/garbage.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights
model = YOLO("Weights/best.pt")

# Define class names
classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > 0.1:
                # Draw red rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle

                # Write "Garbage Detected" on the box
                cv2.putText(img, "Garbage Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()