"""
vector_store.py
A super simple vector store for RAG.

This file just stores documents in memory:
- Each document is converted into an embedding (vector) using OpenAI
- We save both the text and its embedding
- For retrieval, we compute similarity with cosine similarity
"""


# ---------------------------------------------------------------------------
# 3. Vector Store: Using FAISS for similarity search
# ---------------------------------------------------------------------------
# A vector store is like a special database for embeddings. FAISS is a popular
# open-source choice. Here we use `IndexFlatIP` (inner product) + normalization
# to approximate cosine similarity.

from openai_queried_functions import client
from typing import List, Tuple
import openai
import numpy as np


class SimpleVectorStore:
    def __init__(self):
        # List of (text, embedding) pairs
        self.documents = []

    def add_document(self, text: str):
        """Convert text into an embedding and store it."""
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        self.documents.append((text, np.array(embedding)))

    def search(self, query: str, top_k: int = 3):
        """Find the most similar documents to the query."""
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        query_vec = np.array(query_embedding)

        # Compute cosine similarity between query and all docs
        similarities = []
        for doc_text, doc_vec in self.documents:
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((sim, doc_text))

        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in similarities[:top_k]]


# Embedding: Turning text into vectors
# We use `text-embedding-3-small` → cheap, fast, and good enough for a prototype.

EMBED_MODEL = 'text-embedding-3-small'

def get_embedding(text: str) -> List[float]:
    """Call OpenAI API to get embedding vector for a text string."""
    resp = openai.Embedding.create(input=text, model=EMBED_MODEL)
    return resp['data'][0]['embedding']


# Ingest pipeline: Build vector index from raw documents
# Input is a list of (doc_id, text). Each doc is chunked, embedded, and added.

def ingest_documents(doc_texts: List[Tuple[str,str]], store: SimpleVectorStore):
    vectors, metas = [], []
    from embedding_functions import chunk_text
    for doc_id, text in doc_texts:
        chunks = chunk_text(text)
        for i,ch in enumerate(chunks):
            emb = get_embedding(ch)
            vectors.append(emb)
            metas.append({'id': f"{doc_id}__{i}", 'text': ch, 'doc_id': doc_id})
    store.add(vectors, metas)
    return len(vectors)



