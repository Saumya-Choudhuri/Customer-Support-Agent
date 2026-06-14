import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

COLLECTION_NAME = "support_docs"

# Use Ollama's local embedding model — completely free
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"}
    )

def retrieve_context(query: str, n_results: int = 4) -> tuple[str, list[str]]:
    """Return (context_string, list_of_sources)."""
    collection = get_collection()

    if collection.count() == 0:
        return "", []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # Only use results with reasonable similarity (distance < 0.6)
    relevant = [
        (doc, meta["source"], dist)
        for doc, meta, dist in zip(docs, metas, distances)
        if dist < 0.6
    ]

    if not relevant:
        return "", []

    context = "\n\n---\n\n".join([r[0] for r in relevant])
    sources = list(set([r[1] for r in relevant]))

    logger.debug(f"Retrieved {len(relevant)} chunks for: '{query[:50]}'")
    return context, sources