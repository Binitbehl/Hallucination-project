import sys
import pandas as pd
from datasets import load_dataset

from llm import generate_answer, generate_samples, MODEL_NAME
from consistency import consistency_score
from retrieval import KnowledgeBase, grounding_score
from confidence import confidence_score
from labeller import label_answer

# ---------- Settings ----------
N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_SAMPLES = 4
MAX_NEW_TOKENS = 40
PROMPT_SUFFIX = " Answer with one short, direct factual statement. Do not hedge."
OUT_FILE = f"data/results_{N_QUESTIONS}.csv"

# ---------- Data ----------
ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
ds = ds.select(range(N_QUESTIONS))

# Knowledge base: correct reference answers.
# Limitation: easier than real retrieval. State it in your report.
passages = []
for row in ds:
    for ans in row["correct_answers"]:
        if ans.strip() and ans not in passages:
            passages.append(ans)
kb = KnowledgeBase(passages)
print(f"Model: {MODEL_NAME}")
print(f"Knowledge base size: {len(passages)} passages")

# ---------- Main loop ----------
rows = []
for idx, row in enumerate(ds):
    q = row["question"]
    prompt = q + PROMPT_SUFFIX

    main = generate_answer(prompt, max_new_tokens=MAX_NEW_TOKENS)
    samples = generate_samples(prompt, n=N_SAMPLES, max_new_tokens=MAX_NEW_TOKENS)

    cons = consistency_score(main["text"], samples)
    ground, _ = grounding_score([f"{q} {main['text']}"], kb)
    conf = confidence_score(main["token_probs"], main["token_entropies"])
    halluc, c_sup, i_sup = label_answer(
        main["text"], row["correct_answers"], row["incorrect_answers"]
    )

    rows.append({
        "question": q,
        "answer": main["text"],
        "consistency": cons,
        "grounding": ground,
        "confidence": conf,
        "correct_support": c_sup,
        "incorrect_support": i_sup,
        "is_hallucination": halluc,
    })
    print(f"[{idx + 1}/{N_QUESTIONS}] halluc={halluc} cons={cons:.2f} "
          f"ground={ground:.2f} conf={conf:.2f} | {q[:50]}")

    if (idx + 1) % 10 == 0:  # save progress as we go
        pd.DataFrame(rows).to_csv(OUT_FILE, index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)
print(f"\nSaved {OUT_FILE}")
print("Hallucination rate:", round(df["is_hallucination"].mean(), 2))
print("\nAverage signals by label (0 = correct, 1 = hallucinated):")
print(df.groupby("is_hallucination")[["consistency", "grounding", "confidence"]].mean().round(3))