import sys
import pandas as pd
from datasets import load_dataset

from llm import generate_with_context, MODEL_NAME
from retrieval import KnowledgeBase
from labeller import label_answer

MODE = sys.argv[1]            # "loo" or "leaky"
START, END = int(sys.argv[2]), int(sys.argv[3])

res = pd.read_csv(f"data/results_{START}_{END}.csv")
full = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]

passages = []
for row in full:
    for a in row["correct_answers"]:
        if a.strip() and a not in passages:
            passages.append(a)
kb = KnowledgeBase(passages)
print(f"Model: {MODEL_NAME} | mode: {MODE} | rows: {len(res)}")

out = []
for _, r in res.iterrows():
    row = full[int(r["q_index"])]
    q = row["question"]
    own = set(row["correct_answers"])

    hits = [p for p, _ in kb.search(q, k=30 if MODE == "loo" else 3)]
    if MODE == "loo":
        hits = [p for p in hits if p not in own]
    hits = hits[:3]

    if not hits:
        text, status, halluc = "", "abstain", None
    else:
        text = generate_with_context(q, hits)["text"]
        if "don't know" in text.lower() or "do not know" in text.lower():
            status, halluc = "abstain", None
        else:
            status = "answer"
            halluc, _, _ = label_answer(text, row["correct_answers"], row["incorrect_answers"])

    out.append({
        "q_index": r["q_index"],
        "trust": (r["consistency"] + r["grounding"] + r["confidence"]) / 3,
        "orig_halluc": r["is_hallucination"],
        "regen_text": text, "regen_status": status, "regen_halluc": halluc,
    })
    print(f"[{len(out)}/{len(res)}] {status:<8} | {q[:60]}")

df = pd.DataFrame(out)
df.to_csv(f"data/mitigation_{MODE}_{START}_{END}.csv", index=False)

# ---------- Before / after at several thresholds ----------
print(f"\nBEFORE mitigation: hallucination rate = {df['orig_halluc'].mean():.2f} "
      f"(all {len(df)} answers shown to the user)\n")
print(f"{'threshold':>9} {'flagged':>8} {'abstained':>10} {'halluc rate':>12} {'correct answers kept':>21}")
for q in (0.2, 0.4, 0.6):
    thr = df["trust"].quantile(q)
    final_h, answered, correct = [], 0, 0
    for _, d in df.iterrows():
        if d["trust"] >= thr:
            h = d["orig_halluc"]
        elif d["regen_status"] == "abstain":
            continue
        else:
            h = d["regen_halluc"]
        final_h.append(h)
        correct += int(h == 0)
    flagged = int((df["trust"] < thr).sum())
    rate = sum(final_h) / len(final_h) if final_h else float("nan")
    print(f"{thr:>9.2f} {flagged:>8} {len(df) - len(final_h):>10} {rate:>12.2f} {correct:>21}")