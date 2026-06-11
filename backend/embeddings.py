import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

# Load the embedding model once at startup (runs locally, no API needed)
# all-MiniLM-L6-v2 is fast, lightweight, and great for semantic search
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB (persistent local storage)
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(session_id: str):
    """Get or create a ChromaDB collection for a session."""
    collection_name = f"session_{session_id}"
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def embed_and_store_chunks(chunks: List[Dict], session_id: str):
    """
    Generate embeddings for chunks and store in ChromaDB.
    """
    collection = get_or_create_collection(session_id)

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "page_number": chunk["page_number"],
            "filename": chunk["filename"]
        }
        for chunk in chunks
    ]

    # Generate embeddings locally using sentence-transformers
    embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=False).tolist()

    # Store in ChromaDB in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    return len(chunks)


def retrieve_relevant_chunks(query: str, session_id: str, top_k: int = 5) -> List[Dict]:
    """
    Embed the query and retrieve the most relevant chunks from ChromaDB.
    """
    collection = get_or_create_collection(session_id)

    # Check if collection has documents
    if collection.count() == 0:
        return []

    query_embedding = EMBEDDING_MODEL.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page_number": results["metadatas"][0][i]["page_number"],
            "filename": results["metadatas"][0][i]["filename"],
            "relevance_score": round(1 - results["distances"][0][i], 4)
        })

    return chunks


def delete_session_collection(session_id: str):
    """Clean up a session's vector collection."""
    try:
        chroma_client.delete_collection(f"session_{session_id}")
    except Exception:
        pass
