import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "agent_memory"
EMBED_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768

_client = QdrantClient(host="localhost", port=6333)

# Ensure collection exists at import time
_existing = [c.name for c in _client.get_collections().collections]
if COLLECTION_NAME not in _existing:
    _client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def _embed(text: str) -> list:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text).embedding


def store_turn(user_text: str, assistant_text: str):
    combined = f"user: {user_text}\nassistant: {assistant_text}"
    vector = _embed(combined)
    _client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"user": user_text, "assistant": assistant_text},
            )
        ],
    )


def recall(user_text: str, top_k: int = 3) -> list:
    if _client.count(collection_name=COLLECTION_NAME).count == 0:
        return []
    vector = _embed(user_text)
    results = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
    ).points
    return [
        f"user: {r.payload['user']}\nassistant: {r.payload['assistant']}"
        for r in results
    ]
