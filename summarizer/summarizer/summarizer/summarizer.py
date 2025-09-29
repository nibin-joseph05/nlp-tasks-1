from transformers import pipeline
import nltk
nltk.download('punkt')

_summarizer = None

def get_pipeline():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer

def abstractive_summarize(text, min_len=30, max_len=120):
    pipe = get_pipeline()
    summary = pipe(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

