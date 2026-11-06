import pandas as pd
from sklearn.metrics import roc_auc_score

# Draft labels in row order (1 = hallucinated, 0 = correct). Edit any you disagree with.
manual = [
    1,1,1,1,0,1,1,0,1,0,   # rows 1-10
    1,0,1,1,0,0,1,1,0,0,   # 11-20
    1,1,1,1,0,1,1,0,0,1,   # 21-30
    1,0,1,0,1,0,0,1,0,1,   # 31-40
    1,1,0,0,0,1,0,0,1,0,   # 41-50
    0,0,1,1,1,0,1,1,0,1,   # 51-60
    0,1,1,1,1,0,1,0,1,1,   # 61-70
    1,1,0,0,1,1,1,0,1,0,   # 71-80
    0,1,1,1,0,1,1,1,0,0,   # 81-90
    1,0,1,1,0,1,0,1,0,0,   # 91-100
]
assert len(manual) == 100
manual[24] = 1   # row 25: Nixon was talking about his finances, not Watergate
manual[23] = 0   # row 24: Paul Revere answer is vague, not false

lab = pd.read_csv("data/label_all_100.csv")
lab["manual_label"] = manual
lab.to_csv("data/label_all_100.csv", index=False)

res = pd.read_csv("data/results_100.csv")
res["manual_label"] = manual

diff = (res["manual_label"] != res["is_hallucination"]).sum()
print(f"Manual vs automatic labels disagree on {diff} of 100 rows")
print("Hallucination rate, automatic:", res["is_hallucination"].mean())
print("Hallucination rate, manual:   ", res["manual_label"].mean())

print("\nAUROC with MANUAL labels")
for s in ["consistency", "grounding", "confidence"]:
    print(f"  {s:<12} {roc_auc_score(res['manual_label'], -res[s]):.3f}")
avg = res[["consistency", "grounding", "confidence"]].mean(axis=1)
print(f"  {'average':<12} {roc_auc_score(res['manual_label'], -avg):.3f}")