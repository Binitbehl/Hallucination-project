from transformers import pipeline

generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

messages = [{"role": "user", "content": "Who was the first person to walk on the Moon?"}]
result = generator(messages, max_new_tokens=60)
print(result[0]["generated_text"][-1]["content"])