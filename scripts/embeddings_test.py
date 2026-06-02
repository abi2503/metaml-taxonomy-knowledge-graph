# scripts/embeddings_test.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.rag_metaml.embeddings import Embedder

def cosine(a, b):
    return float(np.dot(a, b))

if __name__ == "__main__":
    emb = Embedder()

    s1 = "FinTech > Banking > Digital/Virtual Banking > Banking Platforms"
    s2 = "FinTech > Banking > Digital/Virtual Banking > Mobile Banking"
    s3 = "HealthTech > Digital Health Systems > Electronic Health Record (EHR)"

    E = emb.encode([s1, s2, s3])
    c12 = cosine(E[0], E[1])
    c13 = cosine(E[0], E[2])

    print(f"cos(s1,s2) (should be higher): {c12:.3f}")
    print(f"cos(s1,s3) (should be lower):  {c13:.3f}")
    assert c12 > c13, "Expected related banking paths to be closer than cross-domain paths."
    print(" Embeddings sanity check passed.")
