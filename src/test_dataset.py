from datasets import load_dataset

ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
print("Total questions:", len(ds))
print()
for i in range(3):
    print("Q:", ds[i]["question"])
    print("Best answer:", ds[i]["best_answer"])
    print("Correct answers:", ds[i]["correct_answers"][:2])
    print("Incorrect answers:", ds[i]["incorrect_answers"][:2])
    print()