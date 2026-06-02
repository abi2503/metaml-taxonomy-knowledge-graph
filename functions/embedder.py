"""
functions/embedder.py
Imports graph_loader via full package path: backend.graph_loader
"""
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on path so `backend.graph_loader` resolves
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer
from backend.graph_loader import all_area_nodes, get_node_data

MODEL_NAME    = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

class _EmbedIndex:
    def __init__(self):
        self.model      = None
        self.node_ids   = []
        self.path_strs  = []
        self.matrix     = None
        self.ready      = False

    def build(self):
        t0 = time.time()
        print(f"[embedder] Loading '{MODEL_NAME}' ...", file=sys.stderr)
        self.model = SentenceTransformer(MODEL_NAME)
        pairs = all_area_nodes()
        if not pairs:
            raise RuntimeError("[embedder] No area nodes found in graph.")
        self.node_ids  = [p[0] for p in pairs]
        self.path_strs = [p[1] for p in pairs]
        print(f"[embedder] Encoding {len(pairs)} nodes ...", file=sys.stderr)
        vecs = self.model.encode(
            self.path_strs, batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.matrix = vecs.astype("float32")
        self.ready  = True
        print(f"[embedder] Ready — {len(self.node_ids)} nodes, {time.time()-t0:.1f}s", file=sys.stderr)

_INDEX = _EmbedIndex()

def build_index():
    _INDEX.build()


def _ensure_index():
    """Lazy build for serverless (e.g. Vercel) where startup lifespan is skipped."""
    if not _INDEX.ready:
        build_index()


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float, str]]:
    _ensure_index()
    if not query or not query.strip():
        return []
    q_vec = _INDEX.model.encode(
        [query.strip()], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    scores  = (_INDEX.matrix @ q_vec[0]).tolist()
    k       = min(top_k, len(scores))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(_INDEX.node_ids[i], round(float(scores[i]), 4), _INDEX.path_strs[i]) for i in top_idx]

def classify_text(text: str, top_k: int = 3) -> List[dict]:
    raw = semantic_search(text, top_k=top_k)
    if not raw:
        return []
    max_score = raw[0][1]
    results = []
    for node_id, score, path_str in raw:
        data = get_node_data(node_id)
        results.append({
            "node_id":        node_id,
            "score":          score,
            "path":           path_str,
            "area_name":      data.get("name") or node_id,
            "label":          data.get("label") or "area",
            "confidence_pct": round((score / max_score) * 100) if max_score > 0 else 0,
        })
    return results

def index_stats() -> dict:
    return {
        "ready":         _INDEX.ready,
        "model":         MODEL_NAME,
        "indexed_nodes": len(_INDEX.node_ids),
        "embedding_dim": int(_INDEX.matrix.shape[1]) if _INDEX.ready else 0,
    }


# ── UC3: Concept Similarity ────────────────────────────────────────────────────
def get_node_vector(node_id: str):
    """Return the L2-normalised embedding vector for a specific area node_id."""
    _ensure_index()
    try:
        idx = _INDEX.node_ids.index(node_id)
        return _INDEX.matrix[idx]
    except ValueError:
        return None


def compare_nodes(node_id_a: str, node_id_b: str) -> dict:
    """
    UC3: Cosine similarity between two area nodes using their path embeddings.
    Both nodes must be area-type and present in the embedding index.
    Returns cosine_similarity [0-1], similarity_pct, and interpretation label.
    """
    vec_a = get_node_vector(node_id_a)
    vec_b = get_node_vector(node_id_b)

    if vec_a is None or vec_b is None:
        missing = node_id_a if vec_a is None else node_id_b
        raise ValueError(
            f"Node '{missing}' is not in the embedding index. "
            "Only area-type nodes are indexed."
        )

    sim = float(np.dot(vec_a, vec_b))   # both L2-normalised → dot = cosine
    sim = max(0.0, min(1.0, sim))

    idx_a = _INDEX.node_ids.index(node_id_a)
    idx_b = _INDEX.node_ids.index(node_id_b)

    return {
        "node_a":            {"node_id": node_id_a, "path_str": _INDEX.path_strs[idx_a]},
        "node_b":            {"node_id": node_id_b, "path_str": _INDEX.path_strs[idx_b]},
        "cosine_similarity": round(sim, 4),
        "similarity_pct":    round(sim * 100),
        "interpretation":    "High" if sim >= 0.75 else "Medium" if sim >= 0.45 else "Low",
    }


# ── UC5: Coverage Audit ────────────────────────────────────────────────────────
def find_semantic_overlaps(threshold: float = 0.82, max_pairs: int = 50) -> List[dict]:
    """
    UC5: Find area pairs with cosine similarity >= threshold that live in
    DIFFERENT dimensions — these are potential overlaps or near-duplicates.

    Scans the upper triangle of the (N x N) similarity matrix.
    Efficient at our scale (~600 nodes → 600x600 = 360K pairs, takes ~0.1s).

    Returns list of overlap dicts sorted by:
      1. Cross-dimension first (most interesting)
      2. Similarity descending
    """
    _ensure_index()

    from backend.graph_loader import get_neighbors

    sim_matrix = (_INDEX.matrix @ _INDEX.matrix.T)   # (N, N) float32
    n = len(_INDEX.node_ids)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim < threshold:
                continue

            nid_a = _INDEX.node_ids[i]
            nid_b = _INDEX.node_ids[j]

            parents_a = set(get_neighbors(nid_a, depth=1, direction="up"))
            parents_b = set(get_neighbors(nid_b, depth=1, direction="up"))
            same_dim  = bool(parents_a & parents_b)

            pairs.append({
                "node_a":            {"node_id": nid_a, "path_str": _INDEX.path_strs[i]},
                "node_b":            {"node_id": nid_b, "path_str": _INDEX.path_strs[j]},
                "cosine_similarity": round(sim, 4),
                "similarity_pct":    round(sim * 100),
                "same_dimension":    same_dim,
            })

    pairs.sort(key=lambda p: (-int(not p["same_dimension"]), -p["cosine_similarity"]))
    return pairs[:max_pairs]