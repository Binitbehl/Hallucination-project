from llm import generate_answer, generate_samples
from consistency import consistency_score
from retrieval import KnowledgeBase, grounding_score
from confidence import confidence_score
from trust_score import trust_score, decide

passages = [
    "Canberra is the capital city of Australia.",
    "Neil Armstrong was the first person to walk on the Moon in July 1969.",
    "The 1987 Nobel Prize in Physics was awarded to J. Georg Bednorz and K. Alex Müller for superconductivity in ceramic materials.",
    "Richard Feynman received the Nobel Prize in Physics in 1965.",
    "Albert Einstein received the Nobel Prize in Physics in 1921.",
]
kb = KnowledgeBase(passages)

questions = [
    "What is the capital of Australia?",
    "Who won the Nobel Prize in Physics in 1987?",
]

for q in questions:
    prompt = q + " Answer in one short sentence."
    main = generate_answer(prompt, max_new_tokens=40)
    samples = generate_samples(prompt, n=4, max_new_tokens=40)

    cons = consistency_score(main["text"], samples)
    ground, verdicts = grounding_score([main["text"]], kb)
    conf = confidence_score(main["token_probs"], main["token_entropies"])
    trust = trust_score(cons, ground, conf)

    print("Q:", q)
    print("Answer:", main["text"])
    print(f"Consistency={cons:.2f}  Grounding={ground:.2f}  Confidence={conf:.2f}")
    print(f"TRUST={trust:.2f} -> {decide(trust)}")
    print("-" * 60)