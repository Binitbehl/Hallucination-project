from llm import generate_answer, generate_samples

question = "Who was the first person to walk on the Moon?"

main = generate_answer(question)
print("MAIN ANSWER:", main["text"])
print("Number of tokens:", len(main["token_probs"]))
print("Average token probability:", sum(main["token_probs"]) / len(main["token_probs"]))
print("Lowest token probability:", min(main["token_probs"]))

print("\nSAMPLES:")
for i, s in enumerate(generate_samples(question, n=3), 1):
    print(f"{i}.", s)