import os
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
import json
import torch
import cv2
import numpy as np
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# ✅ FORCE LlamaIndex to use local embeddings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model  # ✅ This ensures NO OpenAI is used

# Initialize FastAPI
app = FastAPI()

# --- Load Models ---
yolo_model = YOLO("/Users/devayushrout/Desktop/MedWaste Guardian/backend/models/yolov8_medical_waste_best.pt")
vosk_model = Model("/Users/devayushrout/Desktop/MedWaste Guardian/backend/models/vosk-model-en-in-0.5") 
recognizer = KaldiRecognizer(vosk_model, 16000)  

# --- Load Legal Compliance Index ---
index_dir = "/Users/devayushrout/Desktop/MedWaste Guardian/backend/models/medwaste_index2"
storage_context = StorageContext.from_defaults(persist_dir=index_dir)
index = load_index_from_storage(storage_context)  # ✅ This now uses the local embeddings

# Load LLaMA-3 locally
llm = HuggingFaceLLM(model_name="meta-llama/Llama-3.2-1B-Instruct")

query_engine = index.as_query_engine(llm=llm)

# --- Speech Recognition API ---
@app.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            if recognizer.AcceptWaveform(audio_chunk):
                result = json.loads(recognizer.Result())
                await websocket.send_text(result["text"])
    except Exception as e:
        print("Connection closed:", e)

# --- Legal Compliance Query API ---
@app.post("/process-text")
async def process_text(query: str = Form(...)):
    response = query_engine.query(query)
    return {"message": response}

# --- Image Processing API ---
@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = yolo_model(image)  
    detections = results[0].boxes.data.cpu().numpy()

    output = [{"label": "Waste", "confidence": float(d[4])} for d in detections]
    return {"detections": output}
