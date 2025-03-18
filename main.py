import numpy as np
import cv2
from PIL import Image

# Function to calculate HSV limits based on the color
def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    lowerlimit = np.array([hsvC[0][0][0] - 10, 100, 100], dtype=np.uint8)
    upperlimit = np.array([hsvC[0][0][0] + 10, 255, 255], dtype=np.uint8)
    return lowerlimit, upperlimit

# Target color (yellow in BGR)
yellow = [0, 255, 255]

# Try to find the camera
cap = None
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
    else:
        print(f"No camera at index {i}")

# Fallback to video file if no camera found
if not cap or not cap.isOpened():
    cap = cv2.VideoCapture("path_to_video.mp4")

lowerlimit, upperlimit = get_limits(color=yellow)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the target color
    mask = cv2.inRange(hsvImage, lowerlimit, upperlimit)
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        print(bbox)

    # Display the resulting mask
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
