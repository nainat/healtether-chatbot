from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
import time

INDEX_PATH = "healtether.index"
CHUNKS_PATH = "chunks.npy"
OLLAMA_MODEL = "phi3:mini"

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


if not os.path.exists(INDEX_PATH):
    raise RuntimeError("❌ FAISS index not found. Run ingest.py first.")

index = faiss.read_index(INDEX_PATH)
chunks = np.load(CHUNKS_PATH, allow_pickle=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")



@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # -------------------------
        # 1️⃣ Embedding timing
        # -------------------------
        start = time.time()
        query_embedding = embedder.encode(
            [request.query], convert_to_numpy=True
        )
        print("Embedding:", round(time.time() - start, 2), "seconds")

        # -------------------------
        # 2️⃣ FAISS timing
        # -------------------------
        start = time.time()
        _, indices = index.search(
            query_embedding.astype("float32"), k=1
        )
        print("FAISS:", round(time.time() - start, 4), "seconds")

        context = "\n\n".join([chunks[i][:300] for i in indices[0]])

        prompt = f"""
You are HealTether’s support assistant.
Use ONLY the context below.
If answer is missing, say:
"I can only help with information available on the HealTether website."

Context:
{context}

Question:
{request.query}
"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }

        # -------------------------
        # 3️⃣ Ollama timing
        # -------------------------
        start = time.time()
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )
        print("Ollama:", round(time.time() - start, 2), "seconds")

        response.raise_for_status()
        answer = response.json().get("response", "").strip()

        return {"answer": answer}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
