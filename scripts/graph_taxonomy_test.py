# scripts/taxonomy_cli.py
"""
One script to query the taxonomy graph by *name* (domain/dimension/area).

What it does:
1) Builds/loads the taxonomy graph (domain -> dimension -> area -> child area).
2) User types any name (domain/dimension/area).
3) Script finds matching node(s) and lets you pick if multiple matches.
4) Prints:
   - Parents (predecessors)
   - Children (successors)
   - Both (neighbors)
   - AND the full successor tree down to leaf (like:
       Trading
         └─ Digital Trading
              └─ Virtual Trading
                  └─ ... leaf
)

Run:
    python scripts/taxonomy_cli.py

Tip:
- Use "depth" for neighborhood hops (e.g., 1 or 2).
- The tree always prints all successors until leaves (with a safety cap).
"""

import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx  # noqa: E402

from src.rag_metaml.graph import build_taxonomy_graph, get_neighbors  # noqa: E402


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def find_nodes_by_name(
    graph: nx.DiGraph,
    query: str,
    *,
    exact: bool = False,
    limit: int = 25,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Find nodes whose 'name' matches query (exact or substring).
    Returns list of (node_id, node_data).
    """
    q = _norm(query)
    hits: List[Tuple[str, Dict[str, Any]]] = []

    for node_id, data in graph.nodes(data=True):
        name = _norm(data.get("name"))
        if not name:
            continue
        ok = (name == q) if exact else (q in name)
        if ok:
            hits.append((node_id, data))
            if len(hits) >= limit:
                break

    return hits


def node_label_name(graph: nx.DiGraph, node_id: str) -> str:
    data = graph.nodes[node_id]
    label = data.get("label") or "N/A"
    name = data.get("name") or "N/A"
    return f"{node_id} | {label} | {name}"


def print_neighbor_block(
    graph: nx.DiGraph,
    node_id: str,
    *,
    depth: int,
    direction: str,
    max_items: int = 20,
) -> None:
    neighbors = get_neighbors(graph, node_id, depth=depth, direction=direction)
    title = {
        "up": "Parents (up / predecessors)",
        "down": "Children (down / successors)",
        "both": "Neighbors (both directions)",
    }.get(direction, f"Neighbors ({direction})")

    print(f"\n--- {title} | depth={depth} | count={len(neighbors)} ---")
    for n in neighbors[:max_items]:
        print("  -", node_label_name(graph, n))
    if len(neighbors) > max_items:
        print(f"  ... (+{len(neighbors) - max_items} more)")


def build_successor_tree_lines(
    graph: nx.DiGraph,
    root: str,
    *,
    max_nodes: int = 500,
    max_depth: int = 50,
) -> List[str]:
    """
    Build a pretty tree of successors from root down to leaves.
    Uses DFS with cycle protection and safety caps.
    """
    lines: List[str] = []
    visited = set()

    # Stack entries: (node, depth, is_last_child, prefix_string, iterator_state)
    def children_of(n: str) -> List[str]:
        # Successors only (taxonomy direction)
        return sorted(list(graph.successors(n)))

    def fmt(n: str) -> str:
        name = graph.nodes[n].get("name") or "N/A"
        label = graph.nodes[n].get("label") or "N/A"
        return f"{name}  [{label}]"

    # Root line
    lines.append(fmt(root))

    # DFS with explicit stack for control
    stack: List[Tuple[str, int, str]] = [(root, 0, "")]  # (node, depth, prefix)
    visited.add(root)
    count = 1

    while stack:
        node, depth, prefix = stack.pop()
        if depth >= max_depth:
            continue

        kids = children_of(node)
        # Push in reverse so first child prints first
        for i in range(len(kids) - 1, -1, -1):
            child = kids[i]

            # Safety: cap total nodes printed
            if count >= max_nodes:
                lines.append(prefix + "└─ " + f"... (tree truncated at {max_nodes} nodes)")
                return lines

            is_last = (i == len(kids) - 1)
            branch = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")

            # Prevent infinite loops if graph has a cycle (shouldn’t, but safe)
            if child in visited:
                lines.append(prefix + branch + fmt(child) + "  (cycle/seen)")
                continue

            lines.append(prefix + branch + fmt(child))
            visited.add(child)
            count += 1

            stack.append((child, depth + 1, child_prefix))

    return lines


# -----------------------------
# Main CLI
# -----------------------------
def main() -> None:
    graph = build_taxonomy_graph()

    print("\n" + "=" * 80)
    print("Taxonomy Explorer (name -> neighbors + successor tree to leaf)")
    print("=" * 80)
    print(f"Graph loaded: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

    query = input("\nEnter a domain/dimension/area NAME (e.g., 'Trading'): ").strip()
    if not query:
        print("No input. Exiting.")
        return

    exact_in = input("Exact match? [y/N]: ").strip().lower()
    exact = exact_in == "y"

    matches = find_nodes_by_name(graph, query, exact=exact, limit=25)
    if not matches:
        print("\nNo matches found. Try a shorter substring (e.g., 'trade').")
        return

    # If multiple matches, pick one
    print(f"\nFound {len(matches)} match(es):")
    for idx, (node_id, data) in enumerate(matches, start=1):
        print(f"  [{idx}] {node_label_name(graph, node_id)}")

    pick_in = input("\nPick a number to explore (default 1): ").strip()
    pick = int(pick_in) if pick_in.isdigit() else 1
    pick = max(1, min(pick, len(matches)))

    node_id, _ = matches[pick - 1]

    # Neighborhood depth (hops)
    depth_in = input("\nNeighborhood hop depth (default 1): ").strip()
    depth = int(depth_in) if depth_in.isdigit() else 1

    print("\n" + "=" * 80)
    print("Selected Node")
    print("=" * 80)
    print(node_label_name(graph, node_id))

    # Neighbors: parents, children, both
    print_neighbor_block(graph, node_id, depth=depth, direction="up")
    print_neighbor_block(graph, node_id, depth=depth, direction="down")
    print_neighbor_block(graph, node_id, depth=depth, direction="both")

    # Full successor tree down to leaf
    print("\n" + "=" * 80)
    print("Successor Tree (down to leaves)")
    print("=" * 80)
    tree_lines = build_successor_tree_lines(graph, node_id, max_nodes=500, max_depth=50)
    for line in tree_lines:
        print(line)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
