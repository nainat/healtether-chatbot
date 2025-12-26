from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os

INDEX_PATH = "healtether.index"
CHUNKS_PATH = "chunks.npy"
OLLAMA_MODEL = "mistral"

OLLAMA_URL = "http://localhost:11434/api/generate"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

# -----------------------------
# LOAD PERSISTENT VECTOR DB
# -----------------------------
if not os.path.exists(INDEX_PATH):
    raise RuntimeError("❌ FAISS index not found. Run ingest.py first.")

index = faiss.read_index(INDEX_PATH)
chunks = np.load(CHUNKS_PATH, allow_pickle=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    query_embedding = embedder.encode([request.query], convert_to_numpy=True)
    _, indices = index.search(query_embedding.astype("float32"), k=3)

    context = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
You are HealTether’s sales and support assistant.

Rules:
- Answer ONLY using the context below.
- Do NOT use outside knowledge.
- If the answer is not in the context, reply exactly:
  "I can only help with information available on the HealTether website."

Context:
{context}

Question:
{request.query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )

    return {"answer": response.json().get("response", "").strip()}
