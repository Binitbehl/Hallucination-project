import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()


def _build_inputs(question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt")


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

    # Probability the model assigned to each token it actually chose
    token_probs = []
    token_entropies = []
    for step, score in enumerate(out.scores):
        probs = torch.softmax(score[0], dim=-1)
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