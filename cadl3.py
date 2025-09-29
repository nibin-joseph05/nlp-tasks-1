# CADL3 – Named Entity Recognition (NER)

import spacy
import pandas as pd

# -------------------------------
# 2. Load spaCy Model
# -------------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# 3. Sample Text Dataset
# -------------------------------
text = """
Elon Musk is the CEO of SpaceX and Tesla. 
Sundar Pichai leads Google from California. 
Satya Nadella works at Microsoft in Redmond.
"""

# -------------------------------
# 4. Process Text with NER
# -------------------------------
doc = nlp(text)

# Extract all entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("\n--- Entities Found ---")
print(entities)

# -------------------------------
# 5. Extract Persons and Organizations into Structured Table
# -------------------------------
persons = []
organizations = []

for ent in doc.ents:
    if ent.label_ == "PERSON":
        persons.append(ent.text)
    elif ent.label_ == "ORG":
        organizations.append(ent.text)

# Create structured DataFrame
data = {"Person": persons, "Organization": organizations}
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

print("\n--- Structured Information (Person–Organization Table) ---")
print(df)