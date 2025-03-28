from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import uvicorn
import os
import json
import numpy as np
import cv2
import wave
from io import BytesIO
from vosk import Model, KaldiRecognizer
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline
from ultralytics import YOLO
from llama_index.llms.huggingface import HuggingFaceLLM  

# Initialize FastAPI
app = FastAPI()

# --- Global Models ---
yolo_model = None
vosk_model = None
recognizer = None
query_engine = None

def load_models():
    """Lazy load models to avoid unnecessary memory usage."""
    global yolo_model, vosk_model, recognizer, query_engine
    if yolo_model is None:
        yolo_model = YOLO("/Users/devayushrout/Desktop/MedWaste Guardian/WasteClassification/resultsyolov8/yolov8_medical_waste/weights/best.pt")

    if vosk_model is None:
        vosk_model = Model("/Users/devayushrout/Desktop/MedWaste Guardian/stt/vosk-model-en-in-0.5")
        recognizer = KaldiRecognizer(vosk_model, 16000)

    if query_engine is None:
        index_dir = "/Users/devayushrout/Desktop/MedWaste Guardian/Legal Compilance/medwaste_index"
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        llm = HuggingFaceLLM(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            model_kwargs={"cache_dir": "/Users/devayushrout/.cache/huggingface"}
        )
        query_engine = index.as_query_engine(llm=llm)

async def convert_audio_to_wav(audio_bytes):
    """Convert input audio bytes to 16kHz WAV format."""
    try:
        audio_stream = BytesIO(audio_bytes)
        with wave.open(audio_stream, 'rb') as wf:
            if wf.getframerate() != 16000:
                raise ValueError("Audio must be 16kHz sample rate")
        return audio_bytes
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio format. Please upload a 16kHz WAV file.")

@app.post("/stt/")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert Speech to Text using Vosk."""
    load_models()
    try:
        audio_data = await audio.read()
        audio_data = await convert_audio_to_wav(audio_data)
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result())
            return {"transcription": result.get("text", "No text detected")}
        return {"error": "Speech recognition failed"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in STT: {str(e)}")

@app.post("/classify/")
async def classify_waste(image: UploadFile = File(...)):
    """Classify waste using YOLOv8."""
    load_models()
    try:
        image_data = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format")

        results = yolo_model(img)
        waste_classes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                if hasattr(yolo_model, "names"):
                    waste_classes.append(yolo_model.names[cls])
                else:
                    waste_classes.append(f"Class {cls}")

        return {"classified_waste": waste_classes}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in classification: {str(e)}")

@app.post("/legal/")
async def legal_compliance(query: str = Form(...)):
    """Get legal compliance information from RAG system."""
    load_models()
    try:
        response = query_engine.query(query)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in legal compliance: {str(e)}")

@app.get("/")
async def root():
    return {"message": "MedWaste Guardian API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
