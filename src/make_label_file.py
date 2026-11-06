import pandas as pd

df = pd.read_csv("data/results_100.csv")
out = df[["question", "answer", "is_hallucination"]].copy()
out["manual_label"] = ""
out.to_csv("data/label_all_100.csv", index=False)
print("Saved data/label_all_100.csv with", len(out), "rows")