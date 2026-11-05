from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def consistency_score(main_answer, sample_answers):
    """Average cosine similarity between the main answer and each sample.
    Returns a value roughly between 0 and 1 (higher = more consistent)."""
    texts = [main_answer] + sample_answers
    embeddings = embedder.encode(texts, convert_to_tensor=True)

    main_emb = embeddings[0]
    sample_embs = embeddings[1:]

    sims = util.cos_sim(main_emb, sample_embs)[0]
    score = sims.mean().item()
    return max(0.0, min(1.0, score))