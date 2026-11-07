import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/results_merged.csv")
y = df["is_hallucination"].values
signals = ["consistency", "grounding", "confidence"]
avg = -df[signals].mean(axis=1).values
rng = np.random.default_rng(0)

for s in signals:
    single = -df[s].values
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], avg[i]) - roc_auc_score(y[i], single[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"average minus {s:<12} mean {np.mean(diffs):+.3f}   95% CI {lo:+.3f} to {hi:+.3f}")