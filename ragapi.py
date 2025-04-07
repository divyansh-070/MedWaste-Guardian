from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from fastapi.middleware.cors import CORSMiddleware
import os
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# === Set HuggingFace embedding model ===
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# === Load Falcon-RW-1B model and tokenizer manually with offloading support ===
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float32,
    offload_folder=None  # ✅ Disk offloading directory
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# === Set HuggingFace LLM model for response generation ===
Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=4096, 
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
)

# ✅ Define paths
DOCS_DIR = "./docs"
INDEX_PATH = "medwaste_index"

if not os.path.exists(INDEX_PATH):
    print("⚙️ No index found. Creating new index...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_PATH)
else:
    print("🔄 Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage_context)

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Allow frontend on port 5000 to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define request model
class QueryRequest(BaseModel):
    query: str

# ✅ Utility to truncate long queries
MAX_QUERY_TOKENS = 200

def truncate_query(text: str, max_tokens: int = MAX_QUERY_TOKENS) -> str:
    return " ".join(text.split()[:max_tokens])

# ✅ API route for querying
@app.post("/query/")
async def query_rag(request: QueryRequest):
    truncated_query = truncate_query(request.query)
    query_engine = index.as_query_engine(similarity_top_k=1)
    response = query_engine.query(truncated_query)
    response_text = response.response.strip()

    print("🔎 Query:", truncated_query)
    print("🧠 Response:", response_text)

    if not response_text or response_text.lower() in ["none", ""]:
        return {
            "response": (
                "I couldn't find a relevant answer. "
                "Try asking about medical waste topics like disposal or biohazards."
            )
        }

    return {"response": response_text}

# ✅ Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
