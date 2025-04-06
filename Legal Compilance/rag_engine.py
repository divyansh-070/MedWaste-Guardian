from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# LLaMA-3.2B model from Hugging Face
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Quantization config (for 8-bit inference)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

# Load LLM
llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer=tokenizer,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "quantization_config": bnb_config}
)

# Load embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)

# Set globally
Settings.llm = llm
Settings.embed_model = embed_model

# Load RAG index from saved path
def load_rag_index(index_path="./medwaste_index"):
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context, settings=Settings)
    return index

# Main query handler
def query_legal_engine(question):
    index = load_rag_index()
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(question)
    return str(response)
