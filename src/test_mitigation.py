from retrieval import KnowledgeBase
from mitigation import answer_with_mitigation

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
    "Who won the Nobel Prize in Chemistry in 1990?",
]

for q in questions:
    r = answer_with_mitigation(q, kb)
    print("Q:", q)
    print("Action:", r["action"], "| original trust:", round(r["trust"], 2))
    print("Final answer:", r["final"])
    if r["source"]:
        print("Source:", r["source"])
    print("-" * 60)