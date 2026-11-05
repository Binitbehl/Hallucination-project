from retrieval import nli_model, NLI_LABELS

pairs = [
    ("Canberra is the capital city of Australia.", "The capital of Australia is Canberra."),
    ("Canberra is the capital city of Australia.", "The capital of Australia is Sydney."),
    ("Watermelon seeds pass through your digestive system.", "Eating watermelon seeds is harmless."),
]

for premise, hypothesis in pairs:
    probs = nli_model.predict([(premise, hypothesis)], apply_softmax=True)[0]
    print("Premise:   ", premise)
    print("Hypothesis:", hypothesis)
    print({label: round(float(p), 3) for label, p in zip(NLI_LABELS, probs)})
    print()