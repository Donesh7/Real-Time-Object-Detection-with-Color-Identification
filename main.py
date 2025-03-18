from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Function to map HSV values to color names
def map_color(hue, value):
    if value < 50:  # Low value indicates black
        return "Black"
    elif hue < 5 or hue > 167:
        return "Red"
    elif hue < 22:
        return "Orange"
    elif hue < 33:
        return "Yellow"
    elif hue < 78:
        return "Green"
    elif hue < 131:
        return "Blue"
    elif hue < 167:
        return "Purple"
    else:
        return "Unknown"

# Function to detect the dominant color(s) in a region of interest (ROI)
def detect_colors(roi):
    # Apply Gaussian blur to reduce noise
    roi = cv2.GaussianBlur(roi, (15, 15), 0)

    # Convert ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Reshape the ROI to a 2D array of pixels
    pixels = hsv_roi.reshape(-1, 3)

    # Use K-means clustering to find the most dominant colors
    kmeans = KMeans(n_clusters=2)  # Detect up to 2 dominant colors
    kmeans.fit(pixels)

    # Get the colors (cluster centers)
    colors = kmeans.cluster_centers_

    # Map the colors to color names
    color_names = [map_color(color[0], color[2]) for color in colors]

    # Return only unique colors
    unique_colors = list(set(color_names))  # Remove duplicates
    return unique_colors

# Load the YOLOv8 model (automatically downloads pre-trained weights if not available)
model = YOLO("yolov8n.pt")  # Use "yolov8n.pt" for the nano version (smallest and fastest)

# Initialize video capture (use 0 for webcam or provide video file path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Get detection details (bounding boxes, class IDs, etc.)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Loop through each detection
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers

            # Extract the region of interest (ROI)
            roi = frame[y1:y2, x1:x2]

            # Detect colors in the ROI
            if roi.size > 0:  # Check if ROI is valid
                colors = detect_colors(roi)  # Detect dominant colors

                # Show only double colors if they exist, otherwise show single color
                if len(colors) >= 2:
                    color_label = f"Colors: {', '.join(colors[:2])}"  # Show only 2 colors
                else:
                    color_label = f"Color: {colors[0]}"  # Show single color
            else:
                color_label = "Colors: Unknown"

            # Get class name
            class_name = result.names[int(class_id)]

            # Display class name and colors on the frame
            label = f"{class_name} ({color_label})"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Object Detection with Color Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
