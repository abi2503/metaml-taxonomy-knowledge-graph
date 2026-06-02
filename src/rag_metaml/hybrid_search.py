from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer

from .graph_utils import get_neighbors, get_taxonomy_path

DEFAULT_COLLECTION = "taxonomy_nodes"
DEFAULT_CHROMA_DIR = "data/chroma_taxonomy"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _format_seed_debug(
    graph: nx.DiGraph,
    seed_ids: List[str],
    seed_distances: List[float],
    seed_documents: List[str],
) -> List[Dict]:
    """
    Convert raw vector search results into a readable debug list.
    """
    out = []
    for i, node_id in enumerate(seed_ids):
        name = graph.nodes[node_id].get("name") if node_id in graph else None
        label = graph.nodes[node_id].get("label") if node_id in graph else None

        out.append(
            {
                "rank": i + 1,
                "node_id": node_id,
                "label": label,
                "name": name,
                "distance": seed_distances[i],
                "doc_preview": seed_documents[i].replace("\n", " ")[:180] + "...",
            }
        )
    return out


def hybrid_search(
    graph: nx.DiGraph,
    query: str,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
    expand_hops: int = 2,
    direction: str = "both",        # "up" | "down" | "both"
    label_filter: Optional[str] = None,  # "domain" | "dimension" | "area" | None
) -> Dict:
    """
    HYBRID RETRIEVAL

    Step A: Vector recall (semantic search) -> seed nodes
    Step B: Graph expansion around seeds -> candidate set
    Step C: Return candidates with taxonomy paths

    Why this is better than traditional RAG?
    - Vector search alone finds "similar" nodes but can miss structure.
    - Graph expansion adds authoritative context (parents/children).
    - Output is explainable using taxonomy paths.
    """
  
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    embedder = SentenceTransformer(model_name)
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]

    where = {"label": label_filter} if label_filter else None

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where,
        include=["metadatas", "distances", "documents"],
    )

    seed_ids: List[str] = res["ids"][0]
    seed_distances: List[float] = res["distances"][0]
    seed_docs: List[str] = res["documents"][0]

    seeds_debug = _format_seed_debug(graph, seed_ids, seed_distances, seed_docs)

    # -----------------------------
    # B) Graph expansion (structure)
    # -----------------------------
    candidates: Set[str] = set(seed_ids)

    for sid in seed_ids:
        # neighbors within expand_hops in chosen direction
        neigh = get_neighbors(graph, sid, depth=expand_hops, direction=direction)
        candidates.update(neigh)

    enriched: List[Dict] = []
    for node_id in sorted(candidates):
        if node_id not in graph:
            continue
        attrs = graph.nodes[node_id]
        if label_filter and attrs.get("label") != label_filter:
            # optional: keep expansion restricted too
            continue

        enriched.append(
            {
                "node_id": node_id,
                "label": attrs.get("label"),
                "name": attrs.get("name") or node_id,
                "path": get_taxonomy_path(graph, node_id),
            }
        )

    return {
        "query": query,
        "seeds": seeds_debug,
        "candidates": enriched,
        "counts": {
            "seed_count": len(seed_ids),
            "candidate_count": len(enriched),
        },
    }
