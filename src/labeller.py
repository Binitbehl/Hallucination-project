import numpy as np
from retrieval import nli_model, NLI_LABELS

ENT = NLI_LABELS.index("entailment")
CON = NLI_LABELS.index("contradiction")


def _probs(answer, references):
    pairs = [(answer, r) for r in references if r.strip()]
    if not pairs:
        return np.zeros((0, 3))
    return nli_model.predict(pairs, apply_softmax=True)


def label_answer(answer, correct_list, incorrect_list):
    """Return (is_hallucination, correct_support, incorrect_support).
    The answer counts as correct only if it clearly supports a correct
    reference and supports it more than any incorrect reference."""
    pc = _probs(answer, correct_list)
    pi = _probs(answer, incorrect_list)

    correct_support = float(pc[:, ENT].max()) if len(pc) else 0.0
    incorrect_support = float(pi[:, ENT].max()) if len(pi) else 0.0
    contradicts_correct = float(pc[:, CON].max()) if len(pc) else 0.0

    is_correct = (
        correct_support > 0.3
        and correct_support > incorrect_support
        and contradicts_correct < 0.5
    )
    return (0 if is_correct else 1), correct_support, incorrect_support