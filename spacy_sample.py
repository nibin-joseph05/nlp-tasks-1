import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

# Download once if needed
# pip install spacy nltk
# python -m spacy download en_core_web_sm
# nltk.download('punkt')
# nltk.download('stopwords')

from nltk.corpus import stopwords

# Initialize spaCy model and stemmer
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
nltk_stopwords = set(stopwords.words("english"))

# Sample text (Kerala news)
text = """On June 7, 2025, the Singapore-flagged container ship MV Wan Hai 503 caught fire following multiple explosions about 88 nautical miles off the coast of Beypore, Kerala. It carries over 2,000 metric tons of fuel and dozens of hazardous material containers. Indian Coast Guard and Navy crews have contained around 40% of the blaze and are towing the vessel away from shore to prevent ecological damage, though 4 crew members remain missing."""

# NLTK RegexpTokenizer (word level)
word_tokenizer = RegexpTokenizer(r'\w+')
words = word_tokenizer.tokenize(text)

# Stemming + Stopword Removal using NLTK
filtered_stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in nltk_stopwords]

print("Words after Stemming & Stopword Removal (NLTK):")
print(filtered_stemmed_words)

# Sentence Tokenization using NLTK Regex
sentence_tokenizer = RegexpTokenizer(r'[^.!?]+[.!?]*')
print("\nSentences (RegexpTokenizer - NLTK):")
print(sentence_tokenizer.tokenize(text))

# Process text with spaCy
doc = nlp(text)

# spaCy word tokens
print("\nWord Tokens (spaCy):")
for token in doc:
    print(token.text)

# spaCy stopword removal
spacy_filtered_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
print("\nWords after Stopword Removal (spaCy):")
print(spacy_filtered_words)

# spaCy sentence segmentation
print("\nSentence Tokens (spaCy):")
for sent in doc.sents:
    print(sent.text)
