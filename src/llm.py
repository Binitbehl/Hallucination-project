import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
if device == "cuda":
    model = model.half()
model.to(device)
model.eval()
print(f"[llm] {MODEL_NAME} on {device}")


def _build_inputs(question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def generate_answer(question, temperature=0.0, max_new_tokens=80):
    """Generate one answer. Returns the text and per-token probabilities."""
    inputs = _build_inputs(question)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    new_tokens = out.sequences[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    token_probs = []
    token_entropies = []
    for step, score in enumerate(out.scores):
        probs = torch.softmax(score[0].float(), dim=-1)
        chosen = new_tokens[step]
        token_probs.append(probs[chosen].item())
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        token_entropies.append(entropy)

    return {
        "text": text,
        "token_probs": token_probs,
        "token_entropies": token_entropies,
    }


def generate_samples(question, n=5, temperature=0.8, max_new_tokens=80):
    """Generate n random answers for the self-consistency module."""
    return [
        generate_answer(question, temperature, max_new_tokens)["text"]
        for _ in range(n)
    ]


def generate_with_context(question, passages, max_new_tokens=60):
    """Answer using ONLY the given passages."""
    context = "\n".join(f"- {p}" for p in passages)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, reply exactly: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer in one short sentence."
    )
    return generate_answer(prompt, temperature=0.0, max_new_tokens=max_new_tokens)