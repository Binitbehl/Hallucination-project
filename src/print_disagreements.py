import pandas as pd

df = pd.read_csv("data/label_all_100.csv")
bad = df[df["manual_label"] != df["is_hallucination"]]
print(f"{len(bad)} rows where manual and automatic labels differ\n")
for i, r in bad.iterrows():
    print(f"ROW {i + 1} | auto={r.is_hallucination} manual={r.manual_label}")
    print("Q:", r.question)
    print("A:", r.answer)
    print()