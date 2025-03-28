import os
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
import json
import torch
import cv2
import numpy as np
import wave
from io import BytesIO
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM

# Initialize FastAPI
app = FastAPI()

# --- Load Models ---
yolo_model = YOLO("/Users/devayushrout/Desktop/MedWaste Guardian/backend/models/yolov8_medical_waste_best.pt")
vosk_model = Model("/Users/devayushrout/Desktop/MedWaste Guardian/backend/models/vosk-model-en-in-0.5")  # Load Vosk Model
recognizer = KaldiRecognizer(vosk_model, 16000)  # Initialize Speech Recognizer

# Load LLaMA-based legal compliance retriever
index_dir = "/Users/devayushrout/Desktop/MedWaste Guardian/backend/models/medwaste_index"
storage_context = StorageContext.from_defaults(persist_dir=index_dir)
index = load_index_from_storage(storage_context)

llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    model_kwargs={"cache_dir": "/Users/devayushrout/.cache/huggingface"}
)
query_engine = index.as_query_engine(llm=llm)

# --- Real-time Speech-to-Text ---
@app.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time speech recognition."""
    await websocket.accept()
    print("Client connected for speech recognition.")

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            if recognizer.AcceptWaveform(audio_chunk):
                result = json.loads(recognizer.Result())
                await websocket.send_text(result["text"])
    except Exception as e:
        print("Connection closed:", e)

# --- Text Processing (Legal Compliance Query) ---
@app.post("/process-text")
async def process_text(query: str = Form(...)):
    """Retrieves legal compliance information based on text input."""
    response = query_engine.query(query)
    return {"message": response}

# --- Image Processing (Waste Classification) ---
@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """Classifies biomedical waste using YOLOv8."""
    image_data = await file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = yolo_model(image)  # Run YOLO detection
    detections = results[0].boxes.data.cpu().numpy()

    output = [{"label": "Waste", "confidence": float(d[4])} for d in detections]
    return {"detections": output}
