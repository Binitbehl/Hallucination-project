import sys
import time
import streamlit as st
from datasets import load_dataset

sys.path.insert(0, "src")

from llm import generate_answer, generate_samples, generate_with_context, MODEL_NAME
from consistency import consistency_score
from retrieval import KnowledgeBase, grounding_score, nli_model, NLI_LABELS
from confidence import confidence_score

SUFFIX = " Answer with one short, direct factual statement. Do not hedge."

DEMO_PASSAGES = [
    "Canberra is the capital city of Australia.",
    "The 1987 Nobel Prize in Physics was awarded to J. Georg Bednorz and K. Alex Müller for superconductivity in ceramic materials.",
    "Neil Armstrong was the first person to walk on the Moon in July 1969.",
]


@st.cache_resource(show_spinner="Building knowledge base (first run only)...")
def get_kb():
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    passages = list(DEMO_PASSAGES)
    for row in ds:
        for a in row["correct_answers"]:
            if a.strip() and a not in passages:
                passages.append(a)
    return KnowledgeBase(passages)


def evidence_support(text, passages):
    """Highest probability that any retrieved passage entails the answer."""
    probs = nli_model.predict([(p, text) for p in passages], apply_softmax=True)
    return float(probs[:, NLI_LABELS.index("entailment")].max())

import re

def retrieve(question, k=5):
    cands = kb.search(question, k=30)
    terms = set(re.findall(r"\b\d{3,4}\b|\b[A-Z][a-z]{3,}\b", question))
    def bonus(p):
        return sum(1 for t in terms if t.lower() in p.lower())
    ranked = sorted(cands, key=lambda c: c[1] + 0.15 * bonus(c[0]), reverse=True)
    return [p for p, _ in ranked[:k]]

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍")
st.title("Hallucination Detection and Mitigation")
st.caption(f"B.Tech CSE mini project| Model: {MODEL_NAME}")

kb = get_kb()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Trust threshold", 0.30, 0.80, 0.55, 0.01)
    n_samples = st.slider("Samples for consistency", 1, 4, 2)
    st.markdown(
        "**Trust score** = average of consistency, grounding and confidence. "
        "Below the threshold, the answer is regenerated from retrieved evidence. "
        "The new answer is accepted only if the evidence supports it, "
        "otherwise the system abstains."
    )

question = st.text_input("Ask a question", "Who won the Nobel Prize in Physics in 1987?")

if st.button("Check answer") and question.strip():
    t0 = time.time()
    prompt = question + SUFFIX

    with st.spinner("Generating answer..."):
        main = generate_answer(prompt, max_new_tokens=40)
    with st.spinner("Sampling answers for consistency..."):
        samples = generate_samples(prompt, n=n_samples, max_new_tokens=40)

    cons = consistency_score(main["text"], samples)
    ground, verdicts = grounding_score([f"{question} {main['text']}"], kb)
    conf = confidence_score(main["token_probs"], main["token_entropies"])
    trust = (cons + ground + conf) / 3

    st.subheader("Model answer")
    st.write(main["text"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consistency", f"{cons:.2f}")
    c2.metric("Grounding", f"{ground:.2f}")
    c3.metric("Confidence", f"{conf:.2f}")
    c4.metric("Trust", f"{trust:.2f}")
    st.progress(max(0.0, min(1.0, trust)))

    if trust >= threshold:
        st.success(f"ACCEPTED: trust {trust:.2f} is at or above the threshold {threshold:.2f}.")
    else:
        st.warning(f"FLAGGED: trust {trust:.2f} is below the threshold {threshold:.2f}. Trying to fix it...")
        passages = retrieve(question, k=5)
        with st.spinner("Regenerating from retrieved evidence..."):
            regen = generate_with_context(question, passages)["text"]

        if "don't know" in regen.lower() or "do not know" in regen.lower():
            st.error("ABSTAINED: the evidence does not contain the answer, so I won't guess.")
        else:
            support = evidence_support(regen, passages)
            if support >= 0.5:
                st.info(f"REGENERATED from evidence (evidence support {support:.2f}):")
                st.write(regen)
            else:
                st.error(
                    f"ABSTAINED: the regenerated answer was not supported by the evidence "
                    f"(support {support:.2f}), so I won't show it."
                )
        with st.expander("Evidence used"):
            for p in passages:
                st.write("-", p)

    with st.expander("Sampled answers (consistency check)"):
        for s in samples:
            st.write("-", s)
    st.caption(f"Total time: {time.time() - t0:.1f} s (CPU runs are slow because of the extra samples)")