from retrieval import KnowledgeBase

passages = [
    "Canberra is the capital city of Australia.",
    "Neil Armstrong was the first person to walk on the Moon in July 1969.",
    "The 1987 Nobel Prize in Physics was awarded to J. Georg Bednorz and K. Alex Müller for superconductivity in ceramic materials.",
    "Richard Feynman received the Nobel Prize in Physics in 1965.",
    "Albert Einstein received the Nobel Prize in Physics in 1921.",
]

kb = KnowledgeBase(passages)

claims = [
    "The capital of Australia is Canberra.",
    "Richard Feynman and Albert Einstein won the Nobel Prize in Physics in 1987.",
]

for claim in claims:
    print("CLAIM:", claim)
    for passage, score in kb.search(claim, k=2):
        print(f"  {score:.2f} | {passage}")
    print()