import pandas as pd
from datasets import load_dataset
from retrieval import KnowledgeBase, check_claim

ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"].select(range(5))
df = pd.read_csv("data/results.csv")

passages = []
for row in ds:
    for ans in row["correct_answers"]:
        if ans.strip() and ans not in passages:
            passages.append(ans)
kb = KnowledgeBase(passages)

for _, r in df.iterrows():
    v = check_claim(r["answer"], kb)
    print("Q:", r["question"])
    print("Answer:  ", r["answer"])
    print("Evidence:", v["passage"])
    print("Verdict: ", v["label"], round(v["entail_prob"], 3))
    print()