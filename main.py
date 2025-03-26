from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import uvicorn
import os
import json
import numpy as np
from vosk import Model, KaldiRecognizer
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline
import cv2
from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI()

# --- Load YOLOv8 Model for Waste Classification ---
yolo_model = YOLO("/Users/devayushrout/Desktop/MedWaste Guardian/WasteClassification/resultsyolov8/yolov8_medical_waste/weights/best.pt")

# --- Load Vosk Speech-to-Text Model ---
vosk_model_path = "/Users/devayushrout/Desktop/MedWaste Guardian/stt/vosk-model-en-in-0.5"
vosk_model = Model(vosk_model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)

# --- Load Legal Compliance RAG Index ---
index_dir = "/Users/devayushrout/Desktop/MedWaste Guardian/Legal Compilance/medwaste_index"

# Force LlamaIndex to use HuggingFace embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)
from llama_index.core.settings import Settings
Settings.embed_model = embed_model  # Ensure no OpenAI embeddings

# Load the index
storage_context = StorageContext.from_defaults(persist_dir=index_dir)
index = load_index_from_storage(storage_context)
from llama_index.llms.huggingface import HuggingFaceLLM  

# Initialize LLaMA-2 model  
llm = HuggingFaceLLM(model_name="meta-llama/Llama-2-7b-chat-hf")  

query_engine = index.as_query_engine(llm=llm)


@app.post("/stt/")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert Speech to Text using Vosk."""
    try:
        audio_data = await audio.read()
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result())
            return {"transcription": result.get("text", "No text detected")}
        return {"error": "Speech recognition failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in STT: {str(e)}")

@app.post("/classify/")
async def classify_waste(image: UploadFile = File(...)):
    """Classify waste using YOLOv8."""
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
                waste_classes.append(yolo_model.names[cls])

        return {"classified_waste": waste_classes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in classification: {str(e)}")

@app.post("/legal/")
async def legal_compliance(query: str = Form(...)):
    """Get legal compliance information from RAG system."""
    try:
        response = query_engine.query(query)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in legal compliance: {str(e)}")

@app.post("/process/")
async def process_input(
    image: UploadFile = None,
    audio: UploadFile = None,
    query: str = Form(None)
):
    """Process input (speech, image, or text) and return relevant compliance information."""
    result = {}
    
    try:
        if audio:
            audio_result = await speech_to_text(audio)
            result["stt"] = audio_result
            query = audio_result.get("transcription", "")

        if image:
            waste_result = await classify_waste(image)
            result["waste_classification"] = waste_result

        if query:
            legal_result = await legal_compliance(query)
            result["legal_compliance"] = legal_result

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
