# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # For BoW and TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # For similarity comparison
from wordcloud import WordCloud  # For word cloud visualization
import matplotlib.pyplot as plt  # For plotting
import pandas as pd  # For matrix representation

# Define sample documents (corpus)
documents = [
    "Cats sleep on warm beds",
    "Dogs bark at strangers",
    "Cats and dogs live together",
    "Strangers fear barking dogs"
]

# ---------------------- Bag-of-Words Section ----------------------

print("\n🔤 Bag-of-Words (BoW):\n")

# Initialize CountVectorizer for BoW representation
bow_vectorizer = CountVectorizer()

# Transform the documents into a BoW sparse matrix
bow_matrix = bow_vectorizer.fit_transform(documents)

# Convert sparse matrix to dense array
bow_array = bow_matrix.toarray()

# Create DataFrame from the array with word tokens as column names
df_bow = pd.DataFrame(bow_array, columns=bow_vectorizer.get_feature_names_out())

# Display the BoW matrix
print("Bag-of-Words Matrix:\n", df_bow)

# Create frequency dictionary: word → total count across all docs
bow_freq = dict(zip(bow_vectorizer.get_feature_names_out(), bow_array.sum(axis=0)))

# Generate a WordCloud using BoW frequencies
wc_bow = WordCloud(background_color="white")
plt.figure(figsize=(6, 4))
plt.imshow(wc_bow.generate_from_frequencies(bow_freq), interpolation="bilinear")
plt.axis("off")  # Hide axes
plt.title("Bag-of-Words Word Cloud")
plt.show()

# Compute cosine similarity between document vectors (BoW)
bow_similarity = cosine_similarity(bow_matrix)

# Display document similarity matrix based on BoW
print("\n📏 Document Similarity Matrix (BoW):\n")
print(pd.DataFrame(bow_similarity))

# ---------------------- TF-IDF Section ----------------------

print("\n📘 TF-IDF:\n")

# Initialize TfidfVectorizer for TF-IDF representation
tfidf_vectorizer = TfidfVectorizer()

# Transform the documents into a TF-IDF sparse matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Convert sparse matrix to dense array
tfidf_array = tfidf_matrix.toarray()

# Create DataFrame from the array with word tokens as column names
df_tfidf = pd.DataFrame(tfidf_array, columns=tfidf_vectorizer.get_feature_names_out())

# Display the TF-IDF matrix
print("TF-IDF Matrix:\n", df_tfidf)

# Create weight dictionary: word → total TF-IDF weight across all docs
tfidf_weight = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_array.sum(axis=0)))

# Generate a WordCloud using TF-IDF weights
wc_tfidf = WordCloud(background_color="white")
plt.figure(figsize=(6, 4))
plt.imshow(wc_tfidf.generate_from_frequencies(tfidf_weight), interpolation="bilinear")
plt.axis("off")  # Hide axes
plt.title("TF-IDF Word Cloud")
plt.show()

# Compute cosine similarity between document vectors (TF-IDF)
tfidf_similarity = cosine_similarity(tfidf_matrix)

# Display document similarity matrix based on TF-IDF
print("\n📏 Document Similarity Matrix (TF-IDF):\n")
print(pd.DataFrame(tfidf_similarity))
