import os

# Set Hugging Face cache directory to avoid redownloading
os.environ["HF_HOME"] = "/Users/devayushrout/.cache/huggingface"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import uvicorn
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
            model_kwargs={"cache_dir": os.environ["HF_HOME"]}
        )
        query_engine = index.as_query_engine(llm=llm)

