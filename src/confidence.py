import math


def confidence_score(token_probs, token_entropies=None):
    """Combine token probabilities into one score in [0, 1].
    Higher = the model was more certain about its words."""
    if not token_probs:
        return 0.0

    # Signal 1: average log-probability, converted back to a probability scale
    mean_logprob = sum(math.log(p + 1e-12) for p in token_probs) / len(token_probs)
    geo_mean = math.exp(mean_logprob)  # geometric mean of token probabilities

    # Signal 2: the least certain token (a single guess can mean a made-up fact)
    min_prob = min(token_probs)

    # Signal 3: average entropy (lower entropy = more certain), squashed to [0, 1]
    if token_entropies:
        avg_entropy = sum(token_entropies) / len(token_entropies)
        entropy_conf = 1.0 / (1.0 + avg_entropy)
    else:
        entropy_conf = geo_mean

    score = 0.5 * geo_mean + 0.2 * min_prob + 0.3 * entropy_conf
    return max(0.0, min(1.0, score))