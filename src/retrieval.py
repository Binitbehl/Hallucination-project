import re
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


# ---------- Claim splitting ----------
def split_claims(text):
    """Split an answer into sentence-level claims."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = [p for p in parts if len(p.split()) >= 3]
    return claims or [text.strip()]


# ---------- NLI-based claim checking ----------
def check_claim(claim, kb, k=3):
    """Retrieve k passages, run NLI on each (passage = premise, claim = hypothesis).
    Returns the best verdict with the entailment probability."""
    results = kb.search(claim, k=k)
    best = {"label": "neutral", "score": 0.0, "passage": None, "entail_prob": 0.0}

    for passage, _ in results:
        probs = nli_model.predict([(passage, claim)], apply_softmax=True)[0]
        entail_prob = float(probs[NLI_LABELS.index("entailment")])
        contra_prob = float(probs[NLI_LABELS.index("contradiction")])

        if entail_prob > best["entail_prob"]:
            best = {
                "label": "entailment" if entail_prob > 0.5 else "neutral",
                "score": entail_prob,
                "passage": passage,
                "entail_prob": entail_prob,
            }
        if contra_prob > 0.5 and contra_prob > best["entail_prob"]:
            best = {
                "label": "contradiction",
                "score": contra_prob,
                "passage": passage,
                "entail_prob": 0.0,
            }
    return best


def grounding_score(claims, kb, k=3):
    """Average entailment probability across all sentence-level claims.
    Returns (score in [0, 1], list of per-claim verdicts)."""
    all_claims = []
    for c in claims:
        all_claims.extend(split_claims(c))
    if not all_claims:
        return 0.0, []
    verdicts = [check_claim(c, kb, k) for c in all_claims]
    score = sum(v["entail_prob"] for v in verdicts) / len(verdicts)
    return score, verdicts