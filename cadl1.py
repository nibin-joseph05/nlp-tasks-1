# ===============================
# CADL1 – Text Preprocessing
import spacy
import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# -------------------------------
# 3. Sample Text
# -------------------------------
text = "Apple is looking at buying U.K. startup for $1 billion. The company has big plans for AI."

print("Original Text:\n", text)

# -------------------------------
# 4. Tokenization
# -------------------------------
tokens = word_tokenize(text)
print("\n--- Tokenization ---\n", tokens)

# -------------------------------
# 5. Stopword Removal
# -------------------------------
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
print("\n--- After Stopword Removal ---\n", filtered_tokens)

# -------------------------------
# 6. Stemming
# -------------------------------
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in filtered_tokens]
print("\n--- After Stemming ---\n", stemmed)

# -------------------------------
# 7. Lemmatization (WordNet)
# -------------------------------
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
print("\n--- After Lemmatization (WordNet) ---\n", lemmatized)

# -------------------------------
# 8. Lemmatization (spaCy)
# -------------------------------
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("\n--- Lemmatization (spaCy) ---")
for token in doc:
    print(f"{token.text:15} → {token.lemma_}")
