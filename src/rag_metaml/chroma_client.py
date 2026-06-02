import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from .config import settings

# Keep Chroma quiet + consistent
os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")

_CHROMA_SETTINGS = ChromaSettings(
    anonymized_telemetry=False,
)

_client = chromadb.PersistentClient(
    path=settings.chroma_dir,
    settings=_CHROMA_SETTINGS,
)

def get_chroma_client() -> chromadb.ClientAPI:
    return _client
