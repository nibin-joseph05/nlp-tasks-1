import spacy
import nltk
from nltk.tokenize import RegexpTokenizer

# Install dependencies (run once):
# pip install spacy nltk
# python -m spacy download en_core_web_sm

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text (recent Kerala news)
text = """On June 7, 2025, the Singapore-flagged container ship MV Wan Hai 503 caught fire following multiple explosions about 88 nautical miles off the coast of Beypore, Kerala. It carries over 2,000 metric tons of fuel and dozens of hazardous material containers. Indian Coast Guard and Navy crews have contained around 40% of the blaze and are towing the vessel away from shore to prevent ecological damage, though 4 crew members remain missing."""

# NLTK RegexpTokenizer: word-level
word_tokenizer = RegexpTokenizer(r'\w+')
print("Words (RegexpTokenizer - NLTK):")
print(word_tokenizer.tokenize(text))

# NLTK RegexpTokenizer: sentence-like chunks
sentence_tokenizer = RegexpTokenizer(r'[^.!?]+[.!?]*')
print("\nSentences (RegexpTokenizer - NLTK):")
print(sentence_tokenizer.tokenize(text))

# Process text with spaCy
doc = nlp(text)

# spaCy word tokens
print("\nWord Tokens (spaCy):")
for token in doc:
    print(token.text)

# spaCy sentence segmentation
print("\nSentence Tokens (spaCy):")
for sent in doc.sents:
    print(sent.text)
