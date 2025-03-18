
# YOLOv8 Object Detection with Color Detection

This project uses the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model for object detection and OpenCV for real-time video processing. Additionally, it detects dominant colors in detected objects using K-means clustering.

## Features
- Real-time object detection using YOLOv8.
- Color detection within detected objects using HSV color space and K-means clustering.
- Displays detected objects with their respective colors.

## Tech Stack
- Object Detection: YOLOv8 (Ultralytics)
- Video Processing: OpenCV (Python)
- Color Detection: K-means clustering (Scikit-learn)
- Programming Language: Python
- Libraries: NumPy, Scikit-learn, OpenCV, Ultralytics

## Requirements

Ensure you have the following dependencies installed:
```bash
pip install ultralytics opencv-python numpy scikit-learn
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script to start object detection and color recognition:
```bash
python detect_objects.py
```
Press `q` to exit the video stream.

## Notes
- The model uses the pre-trained `yolov8n.pt` weights.
- The webcam (device index 0) is used by default; modify `cv2.VideoCapture(0)` to use a different camera or video file.
