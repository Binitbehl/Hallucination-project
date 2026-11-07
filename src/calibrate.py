import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/results_merged.csv")
y = 1 - df["is_hallucination"].values  # 1 = correct
df["trust"] = df[["consistency", "grounding", "confidence"]].mean(axis=1)


def ece(y_true, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum():
            total += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return total


cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(f"Rows: {len(df)} | correct rate: {y.mean():.2f}\n")
print(f"{'score':<12} {'ECE before':>11} {'ECE after':>10} {'AUROC':>7}")
for name in ["confidence", "trust"]:
    raw = df[name].values
    cal = cross_val_predict(
        LogisticRegression(), raw.reshape(-1, 1), y, cv=cv, method="predict_proba"
    )[:, 1]
    print(f"{name:<12} {ece(y, raw):>11.3f} {ece(y, cal):>10.3f} {roc_auc_score(y, raw):>7.3f}")