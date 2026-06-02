from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .db import fetch_areas, fetch_dimensions, fetch_domains, get_engine
from .logging_setup import logger

GRAPH_CACHE_PATH = Path("data/taxonomy_graph.gpickle")


def _add_domain_nodes(graph: nx.DiGraph, domains: Iterable[Dict]) -> None:
    for domain in domains:
        dom_id = int(domain["domain_id"])
        graph.add_node(
            f"domain:{dom_id}",
            label="domain",
            name=domain.get("domain_name"),
            raw=domain,
        )


def _add_dimension_nodes(
    graph: nx.DiGraph, dimensions: Iterable[Dict]
) -> Dict[int, str]:
    """
    Adds dimension nodes.
    Returns a mapping: dimension_id -> dimension_node_id (e.g., 7 -> "dimension:7")
    """
    dim_id_to_node: Dict[int, str] = {}


    for dimension in dimensions:
        try:
            dim_id = int(dimension["dimension_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Bad/missing dimension_id in row: %r", dimension)
            continue

        dim_name = dimension.get("dimension_name")
        dim_node_id = f"dimension:{dim_id}"

        graph.add_node(
            dim_node_id,
            label="dimension",
            name=dim_name,
            raw=dimension,
        )

        dim_id_to_node[dim_id] = dim_node_id

    for dimension in dimensions:
        try:
            dim_id = int(dimension["dimension_id"])
        except (KeyError, TypeError, ValueError):
            continue

        dim_node_id = dim_id_to_node.get(dim_id)
        if not dim_node_id:
            continue

        dom_id = dimension.get("domain_id")
        if dom_id is not None:
            try:
                dom_key = f"domain:{int(dom_id)}"
            except (TypeError, ValueError):
                logger.warning("Dimension %s has invalid domain_id=%r", dim_id, dom_id)
            else:
                if dom_key in graph:
                    graph.add_edge(dom_key, dim_node_id, relation="HAS_DIMENSION")
                else:
                    logger.warning("Dimension %s references missing domain %r", dim_id, dom_key)


        parent_dim = (
            dimension.get("dimension_parent_id")
            or dimension.get("parent_dimension_id")
            or dimension.get("DIMENSION_PARENT_ID")
        )
        if parent_dim is not None:
            try:
                parent_dim_id = int(parent_dim)
                parent_dim_node = f"dimension:{parent_dim_id}"
            except (TypeError, ValueError):
                logger.warning("Dimension %s has invalid parent dimension=%r", dim_id, parent_dim)
            else:
                if parent_dim_node not in graph:
                    # placeholder if parent not in list (rare)
                    graph.add_node(parent_dim_node, label="dimension", name=None, raw=None)

                graph.add_edge(parent_dim_node, dim_node_id, relation="HAS_CHILD_DIMENSION")

    return dim_id_to_node


def _add_area_nodes(
    graph: nx.DiGraph,
    areas: Iterable[Dict],
    dim_id_to_node: Dict[int, str],
) -> None:
    for area in areas:
        try:
            area_id = int(area["area_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Bad/missing area_id in row: %r", area)
            continue

        dim_id = area.get("dimension_id")
        if dim_id is None:
            logger.warning("Area %s missing dimension_id", area_id)
            continue

        try:
            dim_id_int = int(dim_id)
        except (TypeError, ValueError):
            logger.warning("Area %s has invalid dimension_id=%r", area_id, dim_id)
            continue

        area_node_id = f"area:{area_id}"
        graph.add_node(
            area_node_id,
            label="area",
            name=area.get("area_name"),
            raw=area,
        )

        dim_node_id = dim_id_to_node.get(dim_id_int)
        if dim_node_id:
            graph.add_edge(dim_node_id, area_node_id, relation="HAS_AREA")
        else:
            logger.warning(
                "Area %s references dimension %s which was not indexed",
                area_id,
                dim_id_int,
            )

        # area parent-child edges: parent_area -> child_area
        parent_area = area.get("area_parent_id")
        if parent_area is not None:
            try:
                parent_node_id = f"area:{int(parent_area)}"
            except (TypeError, ValueError):
                logger.warning("Area %s has invalid area_parent_id=%r", area_id, parent_area)
                continue

            if parent_node_id not in graph:
                graph.add_node(parent_node_id, label="area", name=None, raw=None)

            graph.add_edge(parent_node_id, area_node_id, relation="HAS_CHILD")


def build_taxonomy_graph(cache_path: Optional[Path] = GRAPH_CACHE_PATH) -> nx.DiGraph:
    """
    Build (or load) the taxonomy graph.

    Directed edges point top-down:
        domain -> dimension -> area -> child area
    """
    if cache_path and cache_path.exists():
        logger.info("Loading taxonomy graph cache from %s", cache_path)
        try:
            return nx.read_gpickle(cache_path)
        except (AttributeError, ImportError):
            import pickle
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    logger.info("Building taxonomy graph from database…")
    engine = get_engine()
    domains = fetch_domains(engine)
    dimensions = fetch_dimensions(engine)
    areas = fetch_areas(engine)

    graph = nx.DiGraph()
    _add_domain_nodes(graph, domains)

    dim_id_to_node = _add_dimension_nodes(graph, dimensions)


    _add_area_nodes(graph, areas, dim_id_to_node)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            nx.write_gpickle(graph, cache_path)
        except (AttributeError, ImportError):
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cached taxonomy graph at %s", cache_path)

    logger.info(
        "Built taxonomy graph with %s nodes and %s edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def get_neighbors(
    graph: nx.DiGraph, node_id: str, depth: int = 1, direction: str = "both"
) -> List[str]:
    """
    Return node ids within the given hop depth.

    direction options:
        - "both" (default): predecessors and successors
        - "up": predecessors only
        - "down": successors only
    """
    if node_id not in graph:
        return []

    if depth <= 0:
        return [node_id]

    visited = {node_id}
    layer = {node_id}
    results: List[str] = []

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


def describe_path(graph: nx.DiGraph, source: str, target: str) -> Optional[List[str]]:
    """
    Return the names along the shortest path between two nodes, if reachable.
    """
    if source not in graph or target not in graph:
        return None

    try:
        path_nodes = nx.shortest_path(graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return None

    return [graph.nodes[n].get("name") or n for n in path_nodes]
