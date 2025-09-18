import nltk
from nltk import RegexpParser
from nltk.tree import Tree
from nltk.tokenize import TreebankWordTokenizer

# 🔽 Download required NLTK models for POS tagging
nltk.download('averaged_perceptron_tagger')  # For tagging words with POS tags

# ✅ Use Treebank tokenizer (instead of default sent_tokenize) to avoid punkt_tab error
tokenizer = TreebankWordTokenizer()

# 📝 Sample text
text = "Alice in Wonderland."

# 🔹 Tokenize the sentence into words
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")  # ['Alice', 'in', 'Wonderland', '.']

# 🔹 Apply POS (Part-of-Speech) tagging
tags = nltk.pos_tag(tokens)
print(f"POS Tags: {tags}")
# Output: [('Alice', 'NNP'), ('in', 'IN'), ('Wonderland', 'NNP'), ('.', '.')]

# 🔹 Define chunking rule using regular expressions
# NP = Noun Phrase → Chunk together one or more proper nouns (NNP)
grammar = r"""
    NP: {<NNP>+}  # One or more Proper Nouns (e.g., 'Alice Wonderland')
"""

# 🔹 Create a chunk parser with the rule above
chunker = RegexpParser(grammar)

# 🔹 Apply chunking on POS-tagged sentence
tree = chunker.parse(tags)

# 🌳 Print the chunked tree as text
print("\nChunked Tree:")
print(tree)

# 🌳 Pretty print tree in terminal
tree.pretty_print()  # Visual tree structure
