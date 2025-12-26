from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

URLS = [
    "https://healtether.com/",
    "https://healtether.com/aboutus/",
    "https://healtether.com/consultation/",
    "https://healtether.com/abdm/",
    "https://healtether.com/pricing/",
    "https://blog.healtether.com/",
    "https://healtether.com/contact/",
    "https://clinic.healtether.com/",

]

DATA_PATH = "data.txt"
INDEX_PATH = "healtether.index"
CHUNKS_PATH = "chunks.npy"

# -----------------------------
# SCRAPE WEBSITE
# -----------------------------
all_text = ""

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for url in URLS:
        print(f"Loading {url}")
        page.goto(url)
        page.wait_for_timeout(3000)
        all_text += page.inner_text("body") + "\n"

    browser.close()

with open(DATA_PATH, "w", encoding="utf-8") as f:
    f.write(all_text)

print(" Website text saved")


def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(all_text)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, convert_to_numpy=True)


index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype("float32"))

faiss.write_index(index, INDEX_PATH)
np.save(CHUNKS_PATH, chunks)

print("Persistent vector DB created")
