"""Pre-download models during build so runtime load is instant."""
import nltk
print("Downloading NLTK words corpus...")
nltk.download("words")

from transformers import AutoTokenizer, AutoModelForCausalLM
print("Downloading distilgpt2 tokenizer...")
AutoTokenizer.from_pretrained("distilgpt2")
print("Downloading distilgpt2 model...")
AutoModelForCausalLM.from_pretrained("distilgpt2")

print("Downloading dbmdz/german-gpt2 tokenizer...")
AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
print("Downloading dbmdz/german-gpt2 model...")
AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")

print("All downloads complete.")
