from pathlib import Path
import sys
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

PERSIST_DIR = "data/chroma_taxonomy"
COLLECTION = "taxonomy_nodes"
MODEL = "all-MiniLM-L6-v2"


if __name__ == "__main__":
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION)

    embedder = SentenceTransformer(MODEL)

    query = "i want to know about Supervisory"
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["metadatas", "distances", "documents"]
    )

    print("\nQUERY:", query)
    print("=" * 80)

    for i, node_id in enumerate(res["ids"][0]):
        meta = res["metadatas"][0][i]
        dist = res["distances"][0][i]
        doc_preview = res["documents"][0][i].split("\n")[0:3]
        print(f"\n#{i+1}: {node_id}")
        print(" label:", meta.get("label"), "| name:", meta.get("name"))
        print(" distance:", dist)
        print(" preview:", " | ".join(doc_preview))
