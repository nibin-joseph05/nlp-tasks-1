from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Text samples
texts = [
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
]

# 2. Labels: 1 = Positive, 0 = Negative
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# 4. Build pipeline: TF-IDF + SVM
model = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel='linear')
)

# 5. Train
model.fit(X_train, y_train)

# 6. Predict
predictions = model.predict(X_test)

# 7. Evaluation
print("Classification Report:\n", classification_report(y_test, predictions))
print("Accuracy Score:", accuracy_score(y_test, predictions) * 100 , "%")

# 8. Custom prediction
custom_text = ["The movie was mind-blowing"]
print("Custom Prediction:", model.predict(custom_text))  # Output: [1] or [0]
