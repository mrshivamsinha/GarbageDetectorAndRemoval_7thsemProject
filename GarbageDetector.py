import cv2
import torch
import math
import cvzone
from ultralytics import YOLO

# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Load the image
image_path = "Media/garbage4.jpeg" 
img = cv2.imread(image_path)

# Resize the image to fit the screen (Adjust width and height as needed)
scale_percent = 50  # Scale down to 50% of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Perform object detection
results = yolo_model(img)

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
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorC=(0, 255, 0), t=2)
            cvzone.putTextRect(img, 'Garbage Detected', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(0, 255, 0))

# Show the resized image
cv2.imshow("Garbage Detection", img)

# Close window when 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
