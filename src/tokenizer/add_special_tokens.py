from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")

tok.add_special_tokens([
    "<|instruction|>",
    "<|input|>",
    "<|output|>",
    "<|system|>"
])

tok.save("tokenizer.json")
print("Added special tokens successfully.")

# run 
# python src/tokenizer/add_special_tokens.py