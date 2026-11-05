from llm import generate_answer, generate_samples, generate_with_context
from consistency import consistency_score
from retrieval import grounding_score
from confidence import confidence_score
from trust_score import trust_score

THRESHOLD = 0.6


def score_answer(question, kb, n_samples=4):
    """Generate an answer and compute all signals and the trust score."""
    prompt = question + " Answer in one short sentence."
    main = generate_answer(prompt, max_new_tokens=40)
    samples = generate_samples(prompt, n=n_samples, max_new_tokens=40)

    cons = consistency_score(main["text"], samples)
    ground, verdicts = grounding_score([main["text"]], kb)
    conf = confidence_score(main["token_probs"], main["token_entropies"])
    trust = trust_score(cons, ground, conf)
    return {"answer": main["text"], "trust": trust, "verdicts": verdicts,
            "cons": cons, "ground": ground, "conf": conf}


def answer_with_mitigation(question, kb, threshold=THRESHOLD):
    first = score_answer(question, kb)

    # Case 1: trusted, return as is
    if first["trust"] >= threshold:
        return {"final": first["answer"], "action": "ACCEPTED",
                "trust": first["trust"], "source": None}

    # Case 2: flagged, regenerate using retrieved evidence
    passages = [p for p, _ in kb.search(question, k=3)]
    regen = generate_with_context(question, passages)
    text = regen["text"]

    if "don't know" in text.lower() or "do not know" in text.lower():
        return {"final": "I'm not confident enough to answer this reliably.",
                "action": "ABSTAINED", "trust": first["trust"], "source": None}

    # Re-check the new answer against the knowledge base
    ground, verdicts = grounding_score([text], kb)
    if ground >= 1.0:
        return {"final": text, "action": "REGENERATED_WITH_SOURCE",
                "trust": first["trust"], "source": verdicts[0]["passage"]}

    # Case 3: still unsupported, abstain
    return {"final": "I'm not confident enough to answer this reliably.",
            "action": "ABSTAINED", "trust": first["trust"], "source": None}