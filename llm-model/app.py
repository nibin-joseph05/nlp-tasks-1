from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

# Load pretrained model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":
        text = request.form["text"]
        # Tokenize long input
        inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
