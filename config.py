import os

from dotenv import load_dotenv
load_dotenv()

# LLM
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3.2:3b")

# Embedding & Memory
EMBED_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768
MEMORY_TOP_K = 3
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agent_memory")

# Persistence
HISTORY_FILE = os.getenv("HISTORY_FILE", "conversation_history.json")
