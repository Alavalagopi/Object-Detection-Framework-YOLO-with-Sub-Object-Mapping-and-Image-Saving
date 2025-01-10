# YOLO-Based Object Detection with Hierarchical JSON Output and Sub-Object Image Saving
This project demonstrates the use of the YOLOv11n (You Only Look Once) model for object detection in a video stream. The script captures frames from a video or webcam, detects objects, and organizes the detected objects into a hierarchical JSON structure. Additionally, it identifies parent-child relationships between objects, crops the detected objects, and saves them as images.
## Table of Contents
1) Features
2) Installation
3) Usage
4) Directory Structure
5) Code Description
6) Output

## Features
*) Detects objects in video frames using a YOLO model.

*) Builds a hierarchical JSON structure representing parent-child relationships among detected objects.

*) Saves cropped images of detected objects and sub-objects.

*) Displays bounding boxes and object labels with confidence scores on video frames.

*) Calculates and displays real-time FPS.
# Installation
Install required dependencies
1) Ensure you have Python 3.7+ installed. Install dependencies using pip:
   
pip install -r requirements.txt  
3) Download YOLO model weights

Place your YOLO model weights (yolo11n.pt) in the project directory.
# Directory Structure
yolo-object-detection-hierarchy/
├── sub_object_images/      # Directory for cropped object images  
├── main.py                 # The main Python script  
├── requirements.txt        # Required Python libraries  
├── yolo11n.pt              # YOLO model weights  
└── README.md               # Documentation file  
# Code Description
Initialization:
Loads the YOLO model (yolo11n.pt).
Initializes video capture (supports both video files and webcam).

Object Detection:
Detects objects in each frame.
Filters objects based on a confidence threshold (default: 0.3).

Hierarchical JSON:
Builds a JSON structure representing the parent-child relationships of detected objects.

Object Cropping:
Crops and saves images of both parent and sub-objects.

Visualization:
Displays bounding boxes, labels, and confidence scores for each detected object.
Shows real-time FPS on the video feed.

# Output
JSON Structure:
A JSON object representing detected objects and their hierarchical relationships is printed to the console.

Cropped Images:
Saved in the sub_object_images directory with filenames indicating the object class and ID.

Video Feed:
Annotated video frames with bounding boxes, labels, and FPS are displayed in real time.

























