import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/results_merged.csv")
y = df["is_hallucination"].values
signals = ["consistency", "grounding", "confidence"]
print(f"Rows: {len(df)} | hallucinated: {y.sum()} | correct: {(1 - y).sum()}\n")


def ci(y, s, n=1000):
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def rep(name, s):
    lo, hi = ci(y, s)
    print(f"{name:<42} AUROC = {roc_auc_score(y, s):.3f}  (95% CI {lo:.2f} to {hi:.2f})")


print("SINGLE SIGNALS")
for s in signals:
    rep(s, -df[s].values)
print("\nSIMPLE AVERAGE")
rep("average of all three", -df[signals].mean(axis=1).values)

print("\nLEARNED COMBINATIONS (5-fold cross-validated)")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
for r in (2, 3):
    for c in itertools.combinations(signals, r):
        p = cross_val_predict(m, df[list(c)].values, y, cv=cv, method="predict_proba")[:, 1]
        rep(" + ".join(c), p)