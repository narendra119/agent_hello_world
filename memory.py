import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    EMBED_MODEL,
    VECTOR_SIZE,
    MEMORY_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
)

COLLECTION_NAME = QDRANT_COLLECTION

_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

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


def recall(user_text: str, top_k: int = MEMORY_TOP_K) -> list:
    if not user_text.strip():
        return []
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
