#!/usr/bin/env python3
"""Run this once to ingest sample docs into ChromaDB."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import chromadb
from chromadb.utils import embedding_functions
import uuid

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'sample_docs')
CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'chroma_db')

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="support_docs",
    embedding_function=ollama_ef,
    metadata={"hnsw:space": "cosine"}
)

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]

print(f"Looking for docs in: {DOCS_DIR}")
found = 0
for filename in os.listdir(DOCS_DIR):
    if not filename.endswith('.md') and not filename.endswith('.txt'):
        continue
    filepath = os.path.join(DOCS_DIR, filename)
    text = open(filepath).read()
    chunks = chunk_text(text)
    if not chunks:
        continue
    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = [{"source": filename} for _ in chunks]
    collection.add(documents=chunks, ids=ids, metadatas=metas)
    print(f"Ingested {filename} -> {len(chunks)} chunks")
    found += 1

if found == 0:
    print("No .md or .txt files found in sample_docs/")
else:
    print(f"\nTotal chunks in collection: {collection.count()}")
    print("Ingestion complete!")
