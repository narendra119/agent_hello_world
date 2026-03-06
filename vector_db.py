import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

EMBEDDING_MODEL = 'nomic-embed-text'
COLLECTION_NAME = 'cat-facts'

# Initialize Qdrant client
client = QdrantClient(host='localhost', port=6333)

# Embedding dimension for nomic-embed-text
EMBEDDING_DIM = 768

def init_collection():
  """Create or reset the collection if it exists"""
  try:
    client.delete_collection(collection_name=COLLECTION_NAME)
  except:
    pass
  
  client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
  )

def create_embedding(text):
  return ollama.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]


def add_chunk_to_database(chunk):
  embedding = create_embedding(chunk)
  # Generate a unique ID based on the chunk content hash
  point_id = hash(chunk) & 0x7FFFFFFF  # Ensure positive integer
  
  # Add point to Qdrant
  client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
      PointStruct(
        id=point_id,
        vector=embedding,
        payload={"text": chunk}
      )
    ]
  )

def retrieve(query, top_n=3):
  query_embedding = create_embedding(query)
  
  # Search in Qdrant using query_points
  search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=top_n
  )
  
  # Extract chunks and scores
  results = [(point.payload["text"], point.score) for point in search_result.points]
  return results
