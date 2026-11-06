import glob
import pandas as pd

files = sorted(glob.glob("data/results_*_*.csv"))
print("Merging:", files)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset="q_index").sort_values("q_index")
df.to_csv("data/results_merged.csv", index=False)
print("Rows:", len(df), "| hallucination rate:", round(df["is_hallucination"].mean(), 2))