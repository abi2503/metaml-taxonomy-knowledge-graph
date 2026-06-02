"""
backend/graph_loader.py
-----------------------
Loaded once at import time. All other modules import GRAPH from here.

Your actual layout:
  RAG_METAML/
  ├── backend/graph_loader.py   <- this file
  ├── data/taxonomy_graph.gpickle
  ├── functions/embedder.py
  └── index/api.py

Pickle resolves to RAG_METAML/data/taxonomy_graph.gpickle automatically.
Override: export GRAPH_CACHE_PATH=/absolute/path/to/your.gpickle
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ── Pickle path ────────────────────────────────────────────────────────────────
# backend/graph_loader.py -> .parent = backend/ -> .parent = RAG_METAML/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_default_pickle = _PROJECT_ROOT / "data" / "taxonomy_graph.gpickle"
GRAPH_CACHE_PATH = Path(os.getenv("GRAPH_CACHE_PATH", str(_default_pickle)))


def _load_graph() -> nx.DiGraph:
    if not GRAPH_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"\n[graph_loader] Pickle not found at: {GRAPH_CACHE_PATH}\n"
            f"Expected: RAG_METAML/data/taxonomy_graph.gpickle\n"
            f"Fix: cp /your/path/taxonomy_graph.gpickle {_PROJECT_ROOT}/data/\n"
            f"  OR: export GRAPH_CACHE_PATH=/absolute/path/to/taxonomy_graph.gpickle"
        )
    try:
        return nx.read_gpickle(str(GRAPH_CACHE_PATH))
    except (AttributeError, ImportError):
        with open(GRAPH_CACHE_PATH, "rb") as f:
            return pickle.load(f)


GRAPH: nx.DiGraph = _load_graph()
print(
    f"[graph_loader] OK — {GRAPH.number_of_nodes()} nodes, "
    f"{GRAPH.number_of_edges()} edges | pickle: {GRAPH_CACHE_PATH}",
    file=sys.stderr,
)


# ── Node helpers ───────────────────────────────────────────────────────────────
def get_node_data(node_id: str) -> Dict[str, Any]:
    if node_id not in GRAPH:
        return {}
    return dict(GRAPH.nodes[node_id])


def get_node_path(node_id: str) -> List[str]:
    """Walk predecessors up to the domain root. Returns name list [domain, dim?, area?]."""
    if node_id not in GRAPH:
        return []
    path_ids = [node_id]
    current = node_id
    for _ in range(10):
        preds = list(GRAPH.predecessors(current))
        if not preds:
            break
        current = preds[0]
        path_ids.insert(0, current)
    return [GRAPH.nodes[n].get("name") or n for n in path_ids]


def get_node_path_str(node_id: str) -> str:
    """Returns 'FinTech > Banking > Banking Platforms'"""
    return " > ".join(get_node_path(node_id))


def get_neighbors(node_id: str, depth: int = 1, direction: str = "both") -> List[str]:
    """
    Return node_ids within `depth` hops.
    direction: 'up' (ancestors) | 'down' (descendants) | 'both'
    """
    if node_id not in GRAPH:
        return []
    visited = {node_id}
    layer = {node_id}
    results: List[str] = []
    for _ in range(depth):
        next_layer = set()
        for n in layer:
            if direction in ("down", "both"):
                next_layer.update(GRAPH.successors(n))
            if direction in ("up", "both"):
                next_layer.update(GRAPH.predecessors(n))
        next_layer.difference_update(visited)
        if not next_layer:
            break
        results.extend(sorted(next_layer))
        visited.update(next_layer)
        layer = next_layer
    return results


def get_subtree(node_id: str, max_nodes: int = 200, max_depth: int = 10) -> List[Dict[str, Any]]:
    """DFS downward, returns structured list for frontend tree rendering."""
    if node_id not in GRAPH:
        return []
    result: List[Dict[str, Any]] = []
    visited = {node_id}
    stack: List[Tuple[str, int, str]] = [(node_id, 0, "")]
    count = 0
    while stack:
        node, depth, prefix = stack.pop()
        if depth >= max_depth:
            continue
        kids = sorted(list(GRAPH.successors(node)))
        for i in range(len(kids) - 1, -1, -1):
            child = kids[i]
            if count >= max_nodes:
                result.append({
                    "id": "__truncated__",
                    "name": f"... ({max_nodes} node limit reached)",
                    "label": "info", "depth": depth + 1,
                    "prefix": prefix, "is_last": True,
                })
                return result
            is_last = (i == len(kids) - 1)
            child_prefix = prefix + ("   " if is_last else "│  ")
            if child in visited:
                continue
            data = GRAPH.nodes[child]
            result.append({
                "id": child,
                "name": data.get("name") or child,
                "label": data.get("label") or "unknown",
                "depth": depth + 1,
                "prefix": prefix,
                "is_last": is_last,
            })
            visited.add(child)
            count += 1
            stack.append((child, depth + 1, child_prefix))
    return result


def search_by_name(query: str, limit: int = 25, exact: bool = False) -> List[Tuple[str, Dict[str, Any]]]:
    q = (query or "").strip().lower()
    hits = []
    for node_id, data in GRAPH.nodes(data=True):
        name = (data.get("name") or "").strip().lower()
        if not name:
            continue
        if (name == q) if exact else (q in name):
            hits.append((node_id, dict(data)))
            if len(hits) >= limit:
                break
    return hits


def all_area_nodes() -> List[Tuple[str, str]]:
    """Returns (node_id, full_path_str) for every area node. Used by embedder."""
    return [
        (nid, get_node_path_str(nid))
        for nid, data in GRAPH.nodes(data=True)
        if data.get("label") == "area"
    ]


def graph_stats() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, data in GRAPH.nodes(data=True):
        lbl = data.get("label") or "other"
        counts[lbl] = counts.get(lbl, 0) + 1
    counts["total_nodes"] = GRAPH.number_of_nodes()
    counts["total_edges"] = GRAPH.number_of_edges()
    return counts