import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/results.csv")

# Hand labels (1 = hallucinated, 0 = correct), in row order. Edit if you disagree.
manual = [1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0]
df["manual_label"] = manual

print("Labels that disagree with automatic labeller:",
      (df["manual_label"] != df["is_hallucination"]).sum(), "of", len(df))
print("Hallucination rate (manual):", df["manual_label"].mean())

# For detection, LOW signal value should mean hallucination, so we negate.
for col in ["consistency", "grounding", "confidence"]:
    auc = roc_auc_score(df["manual_label"], -df[col])
    print(f"AUROC {col}: {auc:.2f}")

combo = (df["consistency"] + df["grounding"] + df["confidence"]) / 3
print(f"AUROC combined (average): {roc_auc_score(df['manual_label'], -combo):.2f}")

df.to_csv("data/results_labeled.csv", index=False)