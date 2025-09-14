"""
ingest.py
Reads documents from the `data/` folder, splits them into chunks,
creates embeddings, and stores them in SimpleVectorStore.
"""

from vector_store import SimpleVectorStore
from PyPDF2 import PdfReader
from typing import List
import os

# Initialize store
store = SimpleVectorStore()

# Where your docs live
DATA_FOLDER = "data"

def read_file(path):
    if path.endswith(".txt") or path.endswith(".md"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif path.endswith(".pdf"):
        text = ""
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        return ""

# Chunking: Breaking long text into smaller pieces
# Why? Embedding models have token limits (e.g., ~8k tokens). Smaller chunks also help retrieval return precise matches instead of whole documents.

def chunk_text(text: str, chunk_size=800, overlap=128) -> List[str]:
    """
    - chunk_size: number of words per chunk
    - overlap: how many words overlap between chunks (keeps context continuity)
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks



def ingest_documents():
    for filename in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, filename)
        text = read_file(path)

        if not text.strip():
            continue  # skip empty docs

        for chunk in chunk_text(text):
            store.add_document(chunk)

    print(f"✅ Ingested {len(store.documents)} chunks into the vector store.")
    return store

if __name__ == "__main__":
    ingest_documents()



