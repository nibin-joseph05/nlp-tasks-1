import nltk
from nltk import RegexpParser
from nltk.tree import Tree

text = "Google is a company. John Doe works there."

tokens = nltk.word_tokenize(text)
print(f"Tokens: {tokens}")

tags = nltk.pos_tag(tokens)
print(f"POS Tags: {tags}")

grammar = r"""
    NP: {<NNP>+}
"""

chunker = RegexpParser(grammar)

tree = chunker.parse(tags)

print("\nChunked Tree:")
print(tree)
tree.pretty_print()

try:
    tree.draw()
except Exception as e:
    print(f"\nCould not draw tree. Make sure tkinter is installed: {e}")