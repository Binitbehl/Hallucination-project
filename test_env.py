import torch, transformers, sentence_transformers, faiss, pandas, sklearn

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("All libraries imported successfully")