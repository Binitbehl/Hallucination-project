import time
import numpy as np
import pandas as pd
from datasets import load_dataset

from llm import generate_answer, generate_samples, MODEL_NAME, device
from consistency import consistency_score
from retrieval import KnowledgeBase, grounding_score
from confidence import confidence_score

N = 10
SUFFIX = " Answer with one short, direct factual statement. Do not hedge."
full = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
passages = list({a for r in full for a in r["correct_answers"] if a.strip()})
kb = KnowledgeBase(passages)

generate_answer("warm up" + SUFFIX, max_new_tokens=5)  # warm-up, not timed

t = {"answer": [], "samples": [], "consistency": [], "grounding": [], "confidence": []}
for i in range(N):
    q = full[i]["question"]
    p = q + SUFFIX
    t0 = time.time(); main = generate_answer(p, max_new_tokens=40); t["answer"].append(time.time() - t0)
    t0 = time.time(); smp = generate_samples(p, n=4, max_new_tokens=40); t["samples"].append(time.time() - t0)
    t0 = time.time(); consistency_score(main["text"], smp); t["consistency"].append(time.time() - t0)
    t0 = time.time(); grounding_score([f"{q} {main['text']}"], kb); t["grounding"].append(time.time() - t0)
    t0 = time.time(); confidence_score(main["token_probs"], main["token_entropies"]); t["confidence"].append(time.time() - t0)

print(f"\nModel: {MODEL_NAME} | device: {device} | questions: {N}")
for k, v in t.items():
    print(f"{k:<12} {np.mean(v):7.3f} s")
base = np.mean(t["answer"])
full_cost = sum(np.mean(v) for v in t.values())
print(f"\nPlain answer only:        {base:.2f} s")
print(f"With all three signals:   {full_cost:.2f} s  ({full_cost / base:.1f}x slower)")