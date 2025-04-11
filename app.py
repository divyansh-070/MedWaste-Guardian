from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
import numpy as np
import cv2
from ultralytics import YOLO
import speech_recognition as sr
import ffmpeg
import os

# === Flask App Setup ===
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# === Embedding Model (MiniLM - fast & light) ===
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# === LLM (Falcon-RW-1B - lightweight and good for local RAG) ===
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-rw-1b",
    torch_dtype=torch.float32,  # or float16 if you're on GPU
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")

Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=2048,
    max_new_tokens=128,
    generate_kwargs={"temperature": 0.5, "do_sample": True},
)

# === Load LlamaIndex RAG Index ===
try:
    storage_context = StorageContext.from_defaults(persist_dir="./medwaste_index")
    index = load_index_from_storage(storage_context)
    print("RAG index loaded successfully.")
except Exception as e:
    print(f"Error loading RAG index: {e}")

# === Load YOLOv8 Model ===
try:
    yolo_model = YOLO("backend/models/yolov8_medical_waste_best.pt")
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# === Truncate Long Input Text ===
def truncate_text(text, max_tokens=200):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

# === Routes ===
@app.route("/")
def home():
    return render_template("text.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        user_query = truncate_text(user_query)
        query_engine = index.as_query_engine(similarity_top_k=1)
        response = query_engine.query(user_query)
        return jsonify({"response": str(response)})
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

import requests

import requests

@app.route("/predict/image", methods=["POST"])
def predict_image():
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image provided"}), 400

        img_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        results = model.predict(img)

        if results and len(results) > 0:
            output = []
            class_names = model.names

            # 📦 Collect predicted object labels
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                conf = float(box.conf[0])

                output.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

            # 🚀 Send RAG query to FastAPI backend
            first_label = output[0]["label"]
            query = f"How should I dispose a {first_label} in biomedical waste management?"

            response = requests.post(
                "http://127.0.0.1:8000/query/",
                json={"query": query}
            )

            rag_answer = response.json().get("response", "No guidance available")

            return jsonify({
                "classification": output,
                "disposal_guidance": rag_answer
            })

        else:
            return jsonify({"classification": "No object detected"})

    except Exception as e:
        return jsonify({"error": f"Image prediction failed: {str(e)}"}), 500



@app.route("/predict/speech", methods=["POST"])
def predict_speech():
    try:
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"error": "No audio provided"}), 400

        webm_path = "temp_audio.webm"
        wav_path = "temp_audio.wav"
        audio_file.save(webm_path)

        ffmpeg.input(webm_path).output(wav_path, format="wav", ar="16000", ac="1").run(overwrite_output=True)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)

        os.remove(webm_path)
        os.remove(wav_path)

        return jsonify({"response": text})
    except Exception as e:
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
