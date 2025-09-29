# lda_example_alt.py

import gensim
from gensim import corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from nltk.corpus import stopwords
import string
import webbrowser
import os

# Download NLTK stopwords only
import nltk
nltk.download('stopwords')

# Sample text corpus
documents = [
    "Artificial Intelligence is transforming the future of technology.",
    "Climate change poses significant challenges for global health.",
    "Advancements in machine learning improve computer vision applications.",
    "Renewable energy sources like solar and wind are crucial for sustainability.",
    "Deep learning methods achieve state-of-the-art results in NLP tasks."
]

# Preprocessing using gensim's simple_preprocess
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = gensim.utils.simple_preprocess(text, deacc=True)  # lowercases + removes punctuation
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

print("🔹 Preprocessed Corpus:", processed_docs)

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Apply LDA
lda_model = gensim.models.LdaModel(
    corpus, num_topics=5, id2word=dictionary, passes=10, random_state=42
)

# Show topics
print("\n🔹 Identified Topics:")
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")

# Visualization
vis = gensimvis.prepare(lda_model, corpus, dictionary)
output_file = "lda_visualization.html"
pyLDAvis.save_html(vis, output_file)

# Auto-open in browser
full_path = os.path.abspath(output_file)
webbrowser.open(f"file://{full_path}")

print("\n✅ LDA visualization opened in your default browser.")
