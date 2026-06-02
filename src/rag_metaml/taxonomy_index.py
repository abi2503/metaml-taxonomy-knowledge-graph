from __future__ import annotations

from typing import Dict, List, Optional
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer

from .node_text import node_to_text

DEFAULT_COLLECTION = "taxonomy_nodes"
DEFAULT_CHROMA_DIR = "data/chroma_taxonomy"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def build_taxonomy_index(
    graph: nx.DiGraph,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    limit: Optional[int] = None,
) -> None:
    """
    Build/update a Chroma vector index for taxonomy nodes.

    What gets indexed?
    - Each node becomes a document using node_to_text()
    - That document is embedded using SentenceTransformer
    - The embedding + metadata is stored in Chroma

    Why PersistentClient?
    - It saves the index to disk so you don’t recompute every run.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    embedder = SentenceTransformer(model_name)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []

    count = 0
    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label")
        if label not in ("domain", "dimension", "area"):
            continue

        doc = node_to_text(graph, node_id)

        ids.append(node_id)
        docs.append(doc)
        metas.append({"label": label, "name": attrs.get("name") or ""})

        count += 1
        if limit and count >= limit:
            break

    # Convert docs -> vectors
    embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()

    # Make it idempotent: remove old ones, then add
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings,
    )

    print(f"✅ Indexed {len(ids)} nodes into Chroma")
    print(f"   persist_dir = {persist_dir}")
    print(f"   collection  = {collection_name}")
    print(f"   model       = {model_name}")
