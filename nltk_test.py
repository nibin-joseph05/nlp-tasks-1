import nltk
from nltk.tokenize import RegexpTokenizer

text = """Nvidia CEO Jensen Huang said that programming artificial intelligence (AI) is similar to how one “programs a person”. Speaking at London Tech Week recently, Huang said that AI is a “great equalizer” as it enables anyone to program using everyday language. Admitting that computing was hard historically, he said “We had to learn programming languages. We had to architect it. We had to design these computers that are very complicated”. “Now, all of a sudden ... there’s a new programming language. This new programming language is called ‘human,’” he added. "Most people don't know C++, very few people know Python, and everybody, as you know, knows human.”"""

# Tokenize into words by matching sequences of word characters
word_tokenizer = RegexpTokenizer(r'\w+')
print("Words (RegexpTokenizer):", word_tokenizer.tokenize(text))

# Tokenize into sentences (this is more complex with regex, often needing
# a pattern that looks for common sentence-ending punctuation followed by a space)
sentence_tokenizer = RegexpTokenizer(r'[^.!?]+[.!?]*')
print("Sentences (RegexpTokenizer):", sentence_tokenizer.tokenize(text))