import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

data = {
    'text': [
        'I love this movie', 
        'This movie is terrible', 
        'I really enjoyed this film', 
        'This film is awful', 
        'What a fantastic experience', 
        'I hated this film', 
        'This was a great movie', 
        'The film was not good', 
        'I am very happy with this movie', 
        'I am disappointed with this film'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Greens')
plt.title("Confusion Matrix - TF-IDF + Naive Bayes")
plt.show()

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
