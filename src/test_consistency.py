from llm import generate_answer, generate_samples
from consistency import consistency_score

questions = [
    "Who was the first person to walk on the Moon?",
    "What is the capital of Australia?",
    "Who won the Nobel Prize in Physics in 1987?",
]

for q in questions:
    main = generate_answer(q, max_new_tokens=40)
    samples = generate_samples(q, n=4, max_new_tokens=40)
    score = consistency_score(main["text"], samples)
    print("Q:", q)
    print("Main:", main["text"])
    print("Consistency score:", round(score, 3))
    print("-" * 60)