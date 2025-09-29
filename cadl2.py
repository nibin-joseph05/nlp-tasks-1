# CADL2 – Bag of Words & TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample dataset (you can replace with movie reviews, tweets, etc.)
corpus = [
    "I just bought a new smartphone and it's super fast.",
    "The laptop battery life is terrible and drains quickly.",
    "This smartwatch is okay, not the best but works fine.",
    "What a fantastic performance by the new gaming console!",
    "The camera quality of this phone is amazing in low light.",
]

print("Corpus:", corpus)

# --- Bag of Words ---
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(corpus)
print("\nBag of Words Vocabulary:\n", vectorizer.get_feature_names_out())
print("\nBoW Representation:\n", X_bow.toarray())

# --- TF-IDF ---
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)
print("\nTF-IDF Vocabulary:\n", tfidf.get_feature_names_out())
print("\nTF-IDF Representation:\n", X_tfidf.toarray())
