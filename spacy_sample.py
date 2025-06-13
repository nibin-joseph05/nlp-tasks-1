import spacy
import nltk
from nltk.tokenize import RegexpTokenizer

# Make sure these are run once to install and download what's needed:
# pip install spacy nltk
# python -m spacy download en_core_web_sm

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = """Nvidia CEO Jensen Huang said that programming artificial intelligence (AI) is similar to how one “programs a person”. Speaking at London Tech Week recently, Huang said that AI is a “great equalizer” as it enables anyone to program using everyday language. Admitting that computing was hard historically, he said “We had to learn programming languages. We had to architect it. We had to design these computers that are very complicated”. “Now, all of a sudden ... there’s a new programming language. This new programming language is called ‘human,’” he added. "Most people don't know C++, very few people know Python, and everybody, as you know, knows human.”"""

# NLTK RegexpTokenizer: word level
word_tokenizer = RegexpTokenizer(r'\w+')
print("Words (RegexpTokenizer - NLTK):")
print(word_tokenizer.tokenize(text))

# NLTK RegexpTokenizer: sentence-like chunks
sentence_tokenizer = RegexpTokenizer(r'[^.!?]+[.!?]*')
print("\nSentences (RegexpTokenizer - NLTK):")
print(sentence_tokenizer.tokenize(text))

# spaCy processing
doc = nlp(text)

# spaCy tokens
print("\nWord Tokens (spaCy):")
for token in doc:
    print(token.text)

# spaCy sentence detection
print("\nSentence Tokens (spaCy):")
for sent in doc.sents:
    print(sent.text)
