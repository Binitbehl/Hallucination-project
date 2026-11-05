from llm import generate_answer
from confidence import confidence_score

questions = [
    "What is the capital of Australia?",
    "Who won the Nobel Prize in Physics in 1987?",
    "Who was the first person to walk on the Moon?",
]

for q in questions:
    out = generate_answer(q, max_new_tokens=40)
    score = confidence_score(out["token_probs"], out["token_entropies"])
    print("Q:", q)
    print("Answer:", out["text"])
    print("Confidence score:", round(score, 3))
    print("-" * 60)