import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------- Models ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

# Label order used by this NLI model's output
NLI_LABELS = ["contradiction", "entailment", "neutral"]


# ---------- Knowledge base ----------
class KnowledgeBase:
    def __init__(self, passages):
        self.passages = passages
        vectors = embedder.encode(passages, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.array(vectors, dtype="float32"))

    def search(self, query, k=3):
        k = min(k, len(self.passages))
        q = embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.array(q, dtype="float32"), k)
        return [(self.passages[i], float(s)) for i, s in zip(ids[0], scores[0])]


# ---------- NLI-based claim checking ----------
def check_claim(claim, kb, k=3):
    """Retrieve k passages, run NLI on each (passage = premise, claim = hypothesis).
    Returns the best verdict: entailment wins over contradiction, which wins over neutral."""
    results = kb.search(claim, k=k)
    best = {"label": "neutral", "score": 0.0, "passage": None}

    for passage, _ in results:
        probs = nli_model.predict([(passage, claim)], apply_softmax=True)[0]
        label_id = int(np.argmax(probs))
        label = NLI_LABELS[label_id]
        conf = float(probs[label_id])

        if label == "entailment" and conf > best["score"]:
            best = {"label": label, "score": conf, "passage": passage}
        elif (
            label == "contradiction"
            and best["label"] != "entailment"
            and conf > best["score"]
        ):
            best = {"label": label, "score": conf, "passage": passage}

    return best


def grounding_score(claims, kb, k=3):
    """Fraction of claims supported (entailed) by the knowledge base.
    Returns (score between 0 and 1, list of per-claim verdicts)."""
    if not claims:
        return 0.0, []
    verdicts = [check_claim(c, kb, k) for c in claims]
    supported = sum(1 for v in verdicts if v["label"] == "entailment")
    return supported / len(claims), verdicts