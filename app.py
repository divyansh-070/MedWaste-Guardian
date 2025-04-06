from flask import Flask, request, jsonify, render_template  # ✅ Include render_template
from flask_cors import CORS
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import speech_recognition as sr
import ffmpeg
import os

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import StorageContext, load_index_from_storage, Settings

# === Set HuggingFace embedding model ===
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# === Load Falcon-RW-1B model + tokenizer with offloading support ===
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device = torch.device("cpu"),
    offload_folder=None  # ✅ Ensure this folder exists
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# === Set the LLM in LlamaIndex Settings ===
Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
)

# === Flask App Setup ===
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# (continue with loading index, endpoints, etc.)


# === Flask Setup ===
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ✅ Load YOLO model
try:
    model = YOLO("backend/models/yolov8_medical_waste_best.pt")
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# ✅ Load LlamaIndex (RAG) with HuggingFace embedding
try:
    storage_context = StorageContext.from_defaults(persist_dir="./medwaste_index")
    index = load_index_from_storage(storage_context)
    print("RAG index loaded successfully.")
except Exception as e:
    print(f"Error loading RAG index: {e}")


# ✅ Serve your frontend
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Make sure templates/index.html exists


# ✂️ Helper: Truncate input to max tokens
def truncate_text(text, max_tokens=200):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

# ✅ Text query route
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        # ⛔ Truncate if too long
        user_query = truncate_text(user_query)

        query_engine = index.as_query_engine(similarity_top_k=1)
        response = query_engine.query(user_query)
        return jsonify({"response": str(response)})
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


# ✅ Image prediction route
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
            data = results[0].boxes.data.cpu().numpy().tolist()
            return jsonify({"classification": data})
        else:
            return jsonify({"classification": "No object detected"})
    except Exception as e:
        return jsonify({"error": f"Image prediction failed: {str(e)}"}), 500


# ✅ Speech-to-text route
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
    app.run(debug=True)
