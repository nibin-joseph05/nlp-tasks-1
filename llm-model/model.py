from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

text = """Artificial intelligence (AI) has become one of the most transformative technologies..."""

inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=30,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

print("Summary:", tokenizer.decode(summary_ids[0], skip_special_tokens=True))
