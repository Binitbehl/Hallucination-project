import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/results_100.csv")
y = df["is_hallucination"].values  # 1 = hallucinated
signals = ["consistency", "grounding", "confidence"]

print(f"Rows: {len(df)} | hallucinated: {y.sum()} | correct: {(1 - y).sum()}\n")


def bootstrap_ci(y, scores, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], scores[idx]))
    return np.percentile(vals, [2.5, 97.5])


def report(name, scores):
    auc = roc_auc_score(y, scores)
    lo, hi = bootstrap_ci(y, scores)
    print(f"{name:<45} AUROC = {auc:.3f}   (95% CI {lo:.2f} to {hi:.2f})")
    return auc


# ---- Single signals (low signal = more likely hallucinated, so we negate) ----
print("SINGLE SIGNALS")
for s in signals:
    report(s, -df[s].values)

# ---- Simple average of all three ----
print("\nSIMPLE AVERAGE")
report("average of all three", -df[signals].mean(axis=1).values)

# ---- Learned combinations (logistic regression, 5-fold cross-validated) ----
print("\nLEARNED COMBINATIONS (cross-validated)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
model = make_pipeline(
    StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
)

results = []
for r in (2, 3):
    for combo in itertools.combinations(signals, r):
        X = df[list(combo)].values
        p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = report(" + ".join(combo), p)
        results.append((" + ".join(combo), auc))

# ---- Which signal matters most (weights on all data) ----
model.fit(df[signals].values, y)
coefs = model.named_steps["logisticregression"].coef_[0]
print("\nLEARNED WEIGHTS (negative = high signal means trustworthy)")
for s, c in zip(signals, coefs):
    print(f"  {s:<12} {c:+.2f}")

# ---- Save 30 random rows for you to hand-check the labeller ----
df.sample(30, random_state=0)[["question", "answer", "is_hallucination"]].to_csv(
    "data/hand_check.csv", index=True
)
print("\nSaved data/hand_check.csv (30 rows to verify by hand)")
df[["question", "answer", "is_hallucination"]].assign(manual_label="").to_csv("data/label_all_100.csv", index=False)