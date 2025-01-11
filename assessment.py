from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import json
import os

# Initialize video capture
cap = cv2.VideoCapture("ppe-2.mp4")
# Uncomment below if using webcam
# cap = cv2.VideoCapture(0)

# Load YOLO model
model = YOLO('yolo11n.pt')

# Initialize FPS variables
prev_frame_time = 0
new_frame_time = 0

# Directory for saving sub-object images
output_dir = "sub_object_images"
os.makedirs(output_dir, exist_ok=True)

# Function to crop and save detected object images
def save_object_image(frame, object_info):
    x1, y1, x2, y2 = map(int, object_info['bbox'])
    cropped_object = frame[y1:y2, x1:x2]
    filename = f"{output_dir}/{object_info['object']}_ID{object_info['id']}.png"
    cv2.imwrite(filename, cropped_object)

while True:
    # Capture frame
    success, img = cap.read()
    if not success:
        print("Failed to capture frame or end of video. Exiting.")
        break

    # Run YOLO prediction
    results = model.predict(img)

    # Hierarchical JSON structure
    hierarchy = []
    parent_objects = []  # List to store parent objects

    # Process YOLO results
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            # Bounding Box Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Filter by confidence threshold
            if conf > 0.3:
                # Class Name
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Determine if the object is a parent or sub-object
                object_info = {
                    "object": class_name,
                    "id": len(hierarchy) + 1,
                    "bbox": [x1, y1, x2, y2],
                    "subobject": []
                }

                # Check spatial relationship to associate sub-objects
                is_sub_object = False
                for parent in parent_objects:
                    px1, py1, px2, py2 = parent["bbox"]
                    # Check if the current object lies within or near a parent object
                    if (x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2) or \
                            (abs(x1 - px1) < w / 2 and abs(y1 - py1) < h / 2):
                        is_sub_object = True
                        parent["subobject"].append(object_info)
                        save_object_image(img, object_info)  # Save sub-object image
                        break

                if not is_sub_object:
                    # If not a sub-object, consider it a parent object
                    parent_objects.append(object_info)
                    hierarchy.append(object_info)
                    save_object_image(img, object_info)  # Save parent object image

                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Convert hierarchy to JSON
    output_json = json.dumps(hierarchy, indent=4)
    print(output_json)  # Print JSON structure to console

    # Display FPS
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=2, thickness=2, colorR=(0, 255, 0))

    # Show frame
    cv2.imshow("Object Detection", img)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
