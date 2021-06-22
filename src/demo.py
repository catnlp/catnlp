from transformers import (
    AutoTokenizer
)

model_path = "resources/data/pretrained/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_path)
text = ["big", "is", "interesting", "and", "multi-processing"]
a = tokenizer(text, is_split_into_words=True)
print(a)
b = tokenizer.decode(a['input_ids'])
print(b)

c = tokenizer.tokenize("interesting")
print(c)
