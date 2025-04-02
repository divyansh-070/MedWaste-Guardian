from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load the YOLO model
model = YOLO("/Users/devayushrout/Desktop/MedWaste Guardian/WasteClassification/yolov8_medical_waste_best.pt")  # Ensure 'best.pt' is in the same directory or provide the full path

app = FastAPI()

@app.post("/detect")
async def detect_waste(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLO model on the image
    results = model(img)

    # Extract detection results
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score
            detections.append({"class": model.names[cls], "confidence": conf})

    return {"detections": detections}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
