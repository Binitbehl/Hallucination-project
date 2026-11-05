import sys
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

from llm import generate_answer, generate_samples
from consistency import consistency_score
from retrieval import KnowledgeBase, grounding_score
from confidence import confidence_score

# ---------- Settings ----------
N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_SAMPLES = 4
MAX_NEW_TOKENS = 40
PROMPT_SUFFIX = " Answer with one short, direct factual statement. Do not hedge."

labeler = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Data ----------
ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
ds = ds.select(range(N_QUESTIONS))

# Knowledge base: correct reference answers
# Limitation: this is easier than real retrieval. State it in your report.
passages = []
for row in ds:
    for ans in row["correct_answers"]:
        if ans.strip() and ans not in passages:
            passages.append(ans)
kb = KnowledgeBase(passages)
print(f"Knowledge base size: {len(passages)} passages")


# ---------- Labeling ----------
def label_answer(answer, correct_list, incorrect_list):
    """Return 1 if the answer looks hallucinated, 0 if it looks correct.
    Automatic approximation: compares the answer to correct vs incorrect references."""
    if not incorrect_list:
        return 0
    a = labeler.encode(answer, convert_to_tensor=True)
    c = labeler.encode(correct_list, convert_to_tensor=True)
    i = labeler.encode(incorrect_list, convert_to_tensor=True)
    sim_c = util.cos_sim(a, c).max().item()
    sim_i = util.cos_sim(a, i).max().item()
    return 0 if sim_c > sim_i else 1


# ---------- Main loop ----------
rows = []
for idx, row in enumerate(ds):
    q = row["question"]
    prompt = q + PROMPT_SUFFIX

    main = generate_answer(prompt, max_new_tokens=MAX_NEW_TOKENS)
    samples = generate_samples(prompt, n=N_SAMPLES, max_new_tokens=MAX_NEW_TOKENS)

    # Signal 1: self-consistency
    cons = consistency_score(main["text"], samples)

    # Signal 2: grounding (the question is included in the claim for context)
    claim_text = f"{q} {main['text']}"
    ground, verdicts = grounding_score([claim_text], kb)

    # Signal 3: confidence
    conf = confidence_score(main["token_probs"], main["token_entropies"])

    # Ground-truth label (automatic approximation)
    halluc = label_answer(main["text"], row["correct_answers"], row["incorrect_answers"])

    rows.append({
        "question": q,
        "answer": main["text"],
        "consistency": cons,
        "grounding": ground,
        "confidence": conf,
        "is_hallucination": halluc,
    })
    print(f"[{idx + 1}/{N_QUESTIONS}] halluc={halluc} cons={cons:.2f} "
          f"ground={ground:.2f} conf={conf:.2f} | {q[:50]}")

# ---------- Save ----------
df = pd.DataFrame(rows)
df.to_csv("data/results.csv", index=False)
print("\nSaved data/results.csv")
print("Hallucination rate:", round(df["is_hallucination"].mean(), 2))
print("\nAverage signals by label (0 = correct, 1 = hallucinated):")
print(df.groupby("is_hallucination")[["consistency", "grounding", "confidence"]].mean().round(3))