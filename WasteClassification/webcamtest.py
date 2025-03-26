import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model (Replace 'best.pt' with your model path)
model = YOLO("/Users/devayushrout/Desktop/MedWaste Guardian/resultsyolov8/yolov8_medical_waste/weights/best.pt")

# Initialize webcam (0 = default webcam, change if using an external camera)
cap = cv2.VideoCapture(0)

# Set webcam width and height
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, conf=0.5)  # Set confidence threshold (adjust as needed)

    # Process results
    for result in results:
        frame_with_boxes = result.plot()  # Draw bounding boxes on frame

    # Display the frame
    cv2.imshow("YOLOv8 Webcam Detection", frame_with_boxes)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


#run this using /Users/devayushrout/Desktop/MedWaste\ Guardian/venv/bin/python webcamtest.py
#as opencv was giving me a hard time
