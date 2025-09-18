# Import TextBlob for NLP tasks
from textblob import TextBlob

# Import detect function to identify the language of text
from langdetect import detect

# ✅ Sample paragraph for analysis
sample_text = """
   Artificial   Intelligence (AI)        and     Data Science    
are       transforming industries rapidly.

   This    blog     explores how     machine    learning models  
  are improving    healthcare, finance,      and education.

The     future looks promising 
      and full    of   exciting      innovations.
"""

# ✅ Create a TextBlob object with the sample text
blob = TextBlob(sample_text)  # This prepares the text for NLP operations

# ✅ Sentiment Analysis
print("=== Sentiment Analysis ===")
print("Polarity     :", blob.sentiment.polarity)        # Value between -1.0 to 1.0 (negative to positive)
print("Subjectivity :", blob.sentiment.subjectivity)    # Value between 0.0 to 1.0 (objective to subjective)

# ✅ Noun Phrase Extraction
print("\n=== Noun Phrases ===")
for phrase in blob.noun_phrases:     # Loops through all detected noun phrases in the text
    print("-", phrase)               # Prints each noun phrase

# ✅ Language Detection using langdetect
try:
    language = detect(sample_text)  # Automatically detects language of the text
    print("\n=== Language Detection ===")
    print("Detected Language:", language)  # Expected output: 'en' (English)
except Exception as e:
    print("\nLanguage detection failed:", e)  # In case detection fails, print the error
