from __future__ import annotations

from typing import List, Optional, Dict, Tuple
import re
import networkx as nx


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace for robust matching."""
    text = text or ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def find_nodes_by_name(graph: nx.DiGraph, name: str, label: Optional[str] = None) -> List[str]:
    """
    Find node_ids whose `name` matches (case-insensitive).
    Optionally filter by label: "domain" | "dimension" | "area"
    """
    target = normalize(name)
    matches: List[str] = []

    for node_id, attrs in graph.nodes(data=True):
        node_name = normalize(attrs.get("name", ""))
        node_label = attrs.get("label")

        if label and node_label != label:
            continue

        if node_name == target:
            matches.append(node_id)

    return matches


def find_nodes_contains(graph: nx.DiGraph, phrase: str, label: Optional[str] = None) -> List[str]:
    """
    Find node_ids where phrase is contained in name.
    Good for quick fuzzy-ish lookup without extra dependencies.
    """
    q = normalize(phrase)
    out: List[str] = []

    for node_id, attrs in graph.nodes(data=True):
        node_name = normalize(attrs.get("name", ""))
        node_label = attrs.get("label")

        if label and node_label != label:
            continue

        if q and q in node_name:
            out.append(node_id)

    return out


def get_parents(graph: nx.DiGraph, node_id: str) -> List[str]:
    """Immediate parents (predecessors)."""
    if node_id not in graph:
        return []
    return list(graph.predecessors(node_id))


def get_children(graph: nx.DiGraph, node_id: str) -> List[str]:
    """Immediate children (successors)."""
    if node_id not in graph:
        return []
    return list(graph.successors(node_id))


def siblings(graph: nx.DiGraph, node_id: str) -> List[str]:
    """
    Nodes that share at least one parent with node_id (excluding itself).
    Useful for "related topics at the same level".
    """
    if node_id not in graph:
        return []

    sibs = set()
    for p in graph.predecessors(node_id):
        for c in graph.successors(p):
            if c != node_id:
                sibs.add(c)
    return sorted(sibs)


def get_taxonomy_path(graph: nx.DiGraph, node_id: str) -> Optional[List[str]]:
    """
    Returns the best-effort taxonomy path names from root domain down to node_id.
    Works best if the graph has:
      domain -> dimension -> area
      and area parent-child edges.

    If multiple paths exist, we pick the shortest path from any domain node.
    """
    if node_id not in graph:
        return None

    # find all domain nodes
    domain_nodes = [n for n, a in graph.nodes(data=True) if a.get("label") == "domain"]
    if not domain_nodes:
        return [graph.nodes[node_id].get("name") or node_id]

    best_path = None
    best_len = None

    for dom in domain_nodes:
        try:
            path_nodes = nx.shortest_path(graph, source=dom, target=node_id)
            if best_len is None or len(path_nodes) < best_len:
                best_path = path_nodes
                best_len = len(path_nodes)
        except nx.NetworkXNoPath:
            continue

    if not best_path:
        return [graph.nodes[node_id].get("name") or node_id]

    # Convert nodes → names
    return [graph.nodes[n].get("name") or n for n in best_path]


def get_subtree(graph: nx.DiGraph, node_id: str, max_depth: int = 10) -> List[str]:
    """
    Returns all descendants reachable from node_id up to max_depth.
    Good for "everything under this dimension/area".
    """
    if node_id not in graph:
        return []

    visited = set([node_id])
    layer = set([node_id])
    out = []

    for _ in range(max_depth):
        nxt = set()
        for n in layer:
            for c in graph.successors(n):
                if c not in visited:
                    nxt.add(c)

        if not nxt:
            break

        out.extend(sorted(nxt))
        visited.update(nxt)
        layer = nxt

    return out

def get_neighbors(
    graph: nx.DiGraph,
    node_id: str,
    depth: int = 1,
    direction: str = "both",
) -> List[str]:
    """
    Return node ids within `depth` hops.

    direction:
        - "down": children only
        - "up": parents only
        - "both": parents + children

    Intuition:
    - This is a controlled BFS over the graph
    - Used for graph expansion in Hybrid RAG
    """
    if node_id not in graph:
        return []

    if depth <= 0:
        return [node_id]

    visited = {node_id}
    layer = {node_id}
    results = []

    for _ in range(depth):
        next_layer = set()

        for node in layer:
            if direction in ("down", "both"):
                next_layer.update(graph.successors(node))
            if direction in ("up", "both"):
                next_layer.update(graph.predecessors(node))

        next_layer.difference_update(visited)
        if not next_layer:
            break

        results.extend(sorted(next_layer))
        visited.update(next_layer)
        layer = next_layer

    return results

