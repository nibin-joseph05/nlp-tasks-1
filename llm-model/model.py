from transformers import BartTokenizer, BartForConditionalGeneration

# Load trained model
tokenizer = BartTokenizer.from_pretrained("./trained_model")
model = BartForConditionalGeneration.from_pretrained("./trained_model")

text = "Large Language Models are changing artificial intelligence research."
inputs = tokenizer([text], max_length=128, truncation=True, return_tensors="pt")

summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=5, length_penalty=2.0)
print("Summary:", tokenizer.decode(summary_ids[0], skip_special_tokens=True))
