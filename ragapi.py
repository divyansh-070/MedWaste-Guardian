from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from fastapi.middleware.cors import CORSMiddleware
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# === ✅ HuggingFace Embedding Model (MiniLM) ===
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# === ✅ Lightweight Falcon-RW-1B LLM ===
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-rw-1b",
    torch_dtype=torch.float32,
    device_map="auto"  # Automatically selects CPU/GPU
)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")

Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=2048,
    max_new_tokens=128,
    generate_kwargs={"temperature": 0.5, "do_sample": True},
)

# === ✅ Index Paths ===
DOCS_DIR = "./data"
INDEX_PATH = "./medwaste_index"

# === ✅ Load or Create Vector Index ===
if not os.path.exists(INDEX_PATH):
    print("📄 No index found. Creating from scratch...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_PATH)
else:
    print("📂 Loading existing RAG index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage_context)

# === ✅ Initialize FastAPI ===
app = FastAPI()

# === ✅ CORS Setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],  # Adjust for frontend port if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ✅ Pydantic Request Model ===
class QueryRequest(BaseModel):
    query: str

# === ✅ Limit Query Length ===
MAX_QUERY_TOKENS = 200

def truncate_query(text: str, max_tokens: int = MAX_QUERY_TOKENS) -> str:
    return " ".join(text.split()[:max_tokens])

# === ✅ API Endpoint ===
@app.post("/query/")
async def query_rag(request: QueryRequest):
    truncated_query = truncate_query(request.query)
    query_engine = index.as_query_engine(similarity_top_k=1)
    response = query_engine.query(truncated_query)
    response_text = str(response).strip()

    print("🔎 Query:", truncated_query)
    print("🧠 Response:", response_text)

    if not response_text or response_text.lower() in ["none", ""]:
        return {
            "response": (
                "I couldn't find a relevant answer. "
                "Try asking about medical waste, segregation, or legal guidelines."
            )
        }

    return {"response": response_text}

# === ✅ Run locally ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
