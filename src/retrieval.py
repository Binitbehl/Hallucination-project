import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")


class KnowledgeBase:
    def __init__(self, passages):
        self.passages = passages
        vectors = embedder.encode(passages, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.array(vectors, dtype="float32"))

    def search(self, query, k=2):
        q = embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.array(q, dtype="float32"), k)
        return [(self.passages[i], float(s)) for i, s in zip(ids[0], scores[0])]