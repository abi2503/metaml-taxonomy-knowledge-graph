"""
index/api.py  —  FastAPI app for MetaML Taxonomy UC1 / UC2 / UC4

CORRECT way to run (from RAG_METAML/ root):
    PYTHONPATH=. uvicorn index.api:app --reload --port 8000

Or use the start script:
    bash start.sh
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # index/ -> RAG_METAML/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import using full package paths — unambiguous regardless of cwd
from backend.graph_loader import (
    GRAPH,
    get_node_data,
    get_node_path,
    get_node_path_str,
    get_neighbors,
    get_subtree,
    search_by_name,
    graph_stats,
)
from functions.embedder import build_index, semantic_search, classify_text, index_stats


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] Building embedding index ...", file=sys.stderr)
    build_index()
    print("[api] Ready.", file=sys.stderr)
    yield


app = FastAPI(
    title="MetaML Taxonomy API",
    version="1.0.0",
    description="UC1: Semantic Search | UC2: Graph Explorer | UC4: Text Classifier",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(3, ge=1, le=10)

class SearchResult(BaseModel):
    node_id: str; score: float; path: str; area_name: str; label: str
    siblings: List[dict] = []

class ClassifyResult(BaseModel):
    node_id: str; score: float; path: str; area_name: str; label: str
    confidence_pct: int; parent_context: List[dict] = []

class NodeDetail(BaseModel):
    node_id: str; name: str; label: str; path: List[str]; path_str: str; raw: dict

class NeighborResult(BaseModel):
    node_id: str; name: str; label: str; path_str: str

class SubtreeNode(BaseModel):
    id: str; name: str; label: str; depth: int; prefix: str; is_last: bool


# ── Helpers ────────────────────────────────────────────────────────────────────
def _to_detail(node_id: str) -> NodeDetail:
    data = get_node_data(node_id)
    if not data:
        raise HTTPException(404, detail=f"Node '{node_id}' not found")
    path = get_node_path(node_id)
    return NodeDetail(
        node_id=node_id, name=data.get("name") or node_id,
        label=data.get("label") or "unknown", path=path,
        path_str=" > ".join(path),
        raw={k: v for k, v in data.items() if k != "raw"},
    )

def _enrich(node_id: str, score: float, path_str: str) -> SearchResult:
    data = get_node_data(node_id)
    siblings = []
    for p in get_neighbors(node_id, depth=1, direction="up")[:1]:
        for sib in get_neighbors(p, depth=1, direction="down"):
            if sib != node_id:
                s = get_node_data(sib)
                siblings.append({"node_id": sib, "name": s.get("name") or sib, "label": s.get("label") or "area"})
    return SearchResult(
        node_id=node_id, score=score, path=path_str,
        area_name=data.get("name") or node_id,
        label=data.get("label") or "area",
        siblings=siblings[:8],
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "graph": graph_stats(), "embedder": index_stats()}

@app.post("/search", response_model=List[SearchResult], tags=["UC1"])
def search(req: SearchRequest):
    raw = semantic_search(req.query, top_k=req.top_k)
    return [_enrich(nid, sc, path) for nid, sc, path in raw]

@app.get("/node/{node_id}", response_model=NodeDetail, tags=["UC2"])
def get_node(node_id: str):
    return _to_detail(node_id)

@app.get("/neighbors/{node_id}", response_model=List[NeighborResult], tags=["UC2"])
def neighbors(
    node_id: str,
    depth: int = Query(1, ge=1, le=5),
    direction: Literal["up", "down", "both"] = Query("both"),
):
    if node_id not in GRAPH:
        raise HTTPException(404, detail=f"Node '{node_id}' not found")
    return [
        NeighborResult(
            node_id=nid, name=get_node_data(nid).get("name") or nid,
            label=get_node_data(nid).get("label") or "unknown",
            path_str=get_node_path_str(nid),
        )
        for nid in get_neighbors(node_id, depth=depth, direction=direction)
    ]

@app.get("/subtree/{node_id}", response_model=List[SubtreeNode], tags=["UC2"])
def subtree(node_id: str, max_nodes: int = Query(150, ge=1, le=500)):
    if node_id not in GRAPH:
        raise HTTPException(404, detail=f"Node '{node_id}' not found")
    return [SubtreeNode(**item) for item in get_subtree(node_id, max_nodes=max_nodes)]

@app.get("/search-name", tags=["UC2"])
def name_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)):
    return [
        {"node_id": nid, "name": d.get("name") or nid,
         "label": d.get("label") or "unknown", "path_str": get_node_path_str(nid)}
        for nid, d in search_by_name(q, limit=limit)
    ]

@app.post("/classify", response_model=List[ClassifyResult], tags=["UC4"])
def classify(req: ClassifyRequest):
    results = []
    for item in classify_text(req.text, top_k=req.top_k):
        nid = item["node_id"]
        ctx = [
            {"node_id": aid, "name": get_node_data(aid).get("name") or aid,
             "label": get_node_data(aid).get("label") or "unknown"}
            for aid in get_neighbors(nid, depth=3, direction="up")
        ]
        results.append(ClassifyResult(
            node_id=nid, score=item["score"], path=item["path"],
            area_name=item["area_name"], label=item["label"],
            confidence_pct=item["confidence_pct"], parent_context=ctx,
        ))
    return results


# ── UC3 models ─────────────────────────────────────────────────────────────────
class CompareRequest(BaseModel):
    node_id_a: str = Field(..., description="First area node_id  e.g. 'area:34'")
    node_id_b: str = Field(..., description="Second area node_id e.g. 'area:235'")


class SimilarityResult(BaseModel):
    node_a: dict
    node_b: dict
    cosine_similarity: float
    similarity_pct: int
    interpretation: str          # "High" | "Medium" | "Low"
    graph_path: List[str]        # shortest path names through graph (may be empty)
    graph_distance: int          # hop count, -1 if unreachable


# ── UC5 models ─────────────────────────────────────────────────────────────────
class OverlapPair(BaseModel):
    node_a: dict
    node_b: dict
    cosine_similarity: float
    similarity_pct: int
    same_dimension: bool


class CoverageStats(BaseModel):
    dimension_name: str
    domain_name: str
    area_count: int


# ── UC3 route ──────────────────────────────────────────────────────────────────
@app.post("/compare", response_model=SimilarityResult, tags=["UC3 - Similarity Mapper"])
def compare(req: CompareRequest):
    """
    UC3: Compare two area nodes.
    Returns cosine similarity from embeddings + shortest graph path between them.
    Both node_ids must be area-type nodes present in the embedding index.
    """
    from functions.embedder import compare_nodes

    try:
        result = compare_nodes(req.node_id_a, req.node_id_b)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # Shortest graph path (undirected view so we can traverse up and across)
    import networkx as nx
    undirected = GRAPH.to_undirected()
    try:
        path_nodes = nx.shortest_path(undirected, req.node_id_a, req.node_id_b)
        graph_path     = [get_node_data(n).get("name") or n for n in path_nodes]
        graph_distance = len(path_nodes) - 1
    except nx.NetworkXNoPath:
        graph_path     = []
        graph_distance = -1

    return SimilarityResult(
        **result,
        graph_path=graph_path,
        graph_distance=graph_distance,
    )


# ── UC5 routes ─────────────────────────────────────────────────────────────────
@app.get("/coverage", response_model=List[CoverageStats], tags=["UC5 - Coverage Audit"])
def coverage():
    """
    UC5: Area count per dimension across the full taxonomy.
    Used to render the coverage bar chart in the frontend.
    """
    stats = []
    for node_id, data in GRAPH.nodes(data=True):
        if data.get("label") != "dimension":
            continue
        dim_name = data.get("name") or node_id
        area_count = sum(
            1 for child in GRAPH.successors(node_id)
            if GRAPH.nodes[child].get("label") == "area"
        )
        # Walk up to find the domain
        domain_name = "Unknown"
        for pred in GRAPH.predecessors(node_id):
            if GRAPH.nodes[pred].get("label") == "domain":
                domain_name = GRAPH.nodes[pred].get("name") or pred
                break

        stats.append(CoverageStats(
            dimension_name=dim_name,
            domain_name=domain_name,
            area_count=area_count,
        ))

    stats.sort(key=lambda s: -s.area_count)
    return stats


@app.get("/overlaps", response_model=List[OverlapPair], tags=["UC5 - Coverage Audit"])
def overlaps(
    threshold: float = Query(0.82, ge=0.5, le=0.99,
                             description="Cosine similarity threshold (0.5–0.99). Higher = stricter."),
    max_pairs: int   = Query(50,   ge=1,   le=200),
):
    """
    UC5: Find semantically similar area pairs (cosine >= threshold) that live in
    different dimensions — potential taxonomy overlaps or near-duplicates.
    Cross-dimension pairs are returned first, sorted by similarity desc.
    """
    from functions.embedder import find_semantic_overlaps
    pairs = find_semantic_overlaps(threshold=threshold, max_pairs=max_pairs)
    return [OverlapPair(**p) for p in pairs]