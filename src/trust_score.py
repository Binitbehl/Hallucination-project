def trust_score(consistency, grounding, confidence,
                w_cons=0.4, w_ground=0.4, w_conf=0.2):
    """Weighted average of the three signals, in [0, 1]. Higher = more trustworthy."""
    return w_cons * consistency + w_ground * grounding + w_conf * confidence


def decide(trust, threshold=0.6):
    return "ACCEPT" if trust >= threshold else "FLAG"