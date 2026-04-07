"""Pre-download models during build so runtime load is instant."""
import nltk
print("Downloading NLTK words corpus...")
nltk.download("words")

from transformers import GPT2Tokenizer, GPT2LMHeadModel
print("Downloading distilgpt2 tokenizer...")
GPT2Tokenizer.from_pretrained("distilgpt2")
print("Downloading distilgpt2 model...")
GPT2LMHeadModel.from_pretrained("distilgpt2")
print("All downloads complete.")
