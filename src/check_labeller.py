import pandas as pd
from datasets import load_dataset
from labeller import label_answer

df = pd.read_csv("data/results_labeled.csv")
ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"].select(range(len(df)))

new_labels = []
for i, row in enumerate(ds):
    lab, _, _ = label_answer(df.loc[i, "answer"], row["correct_answers"], row["incorrect_answers"])
    new_labels.append(lab)

df["nli_label"] = new_labels
old_wrong = (df["is_hallucination"] != df["manual_label"]).sum()
new_wrong = (df["nli_label"] != df["manual_label"]).sum()
print(f"Old embedding labeller wrong: {old_wrong} of {len(df)}")
print(f"New NLI labeller wrong:       {new_wrong} of {len(df)}")
print()
print(df.loc[df["nli_label"] != df["manual_label"], ["question", "answer", "manual_label", "nli_label"]].to_string())