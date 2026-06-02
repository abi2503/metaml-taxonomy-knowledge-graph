# scripts/index_taxonomy_test.py
import chromadb
from chromadb.config import Settings as ChromaSettings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_metaml.index_taxonomy import build_or_refresh_taxonomy_index
from rag_metaml.embeddings import Embedder
from rag_metaml.config import settings
from rag_metaml.chroma_client import get_chroma_client

if __name__ == "__main__":

    n = build_or_refresh_taxonomy_index()
    print(f"Indexed rows: {n}")

    client = get_chroma_client()
    col = client.get_or_create_collection(settings.taxonomy_collection, metadata={"hnsw:space": "cosine"})
    emb = Embedder()

    query = "stable coins"
    qv = emb.encode([query])[0].tolist()

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Query Embedding (first 10 dimensions): {qv[:10]}")
    print(f"Query Embedding dimensions: {len(qv)}")
    print(f"{'='*80}\n")

    res = col.query(
        query_embeddings=[qv],
        n_results=5,
        include=["metadatas", "documents", "distances", "embeddings"]
    )

    if not res or not res.get("metadatas"):
        print("No results — check that the index build succeeded.")
    else:
        print("\nTop matches:")
        for idx, (meta, doc, dist, embedding) in enumerate(zip(
            res["metadatas"][0], 
            res["documents"][0], 
            res["distances"][0],
            res["embeddings"][0]
        ), 1):
            score = 1.0 - float(dist)
            print(f"\n{idx}. [{score:.3f}] {meta.get('DOMAIN_NAME')} / {meta.get('DIMENSION_NAME')} / {meta.get('AREA_NAME')}")
            print(f"    Document: {doc}")
            print(f"    Distance: {dist:.4f}")
            print(f"    Embedding (first 10 dimensions): {embedding[:10]}")
            print(f"    Embedding dimensions: {len(embedding)}")
