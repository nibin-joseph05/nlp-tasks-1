import spacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('stopwords')

# Initialize
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Sample text
text = "Hi Nibin! You're testing NLP preprocessing: stemming, lemmatization, and stopword removal."

# Process with spaCy
doc = nlp(text)

# Results
print("Original Tokens:")
for token in doc:
    print(token.text)

print("\nAfter Preprocessing:")
for token in doc:
    if token.is_punct or token.is_space:
        continue

    word = token.text.lower()

    if word in stop_words:
        continue

    lemma = token.lemma_
    stem = stemmer.stem(word)

    print(f"Word: {word} | Lemma: {lemma} | Stem: {stem}")
