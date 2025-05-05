import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from bert_model import load_bert_model, predict_bert_prob

def train_and_save_model(corpus, labels, progress_callback=None):
    vect = CountVectorizer(max_features=1500)
    X = vect.fit_transform(corpus).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

    if progress_callback:
        progress_callback("start")

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    if progress_callback:
        progress_callback("stop")

    joblib.dump(classifier, "model.joblib")
    joblib.dump(vect, "vect.joblib")

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cm, accuracy

def load_model_and_vectorizer():
    try:
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vect.joblib")
        return model, vectorizer
    except FileNotFoundError as e:
        raise FileNotFoundError("Model or vectorizer file not found.") from e

def predict_sentiment_ensemble(model, vectorizer, bert_model, bert_tokenizer, input_text):
    input_vector = vectorizer.transform([input_text])
    prob_lr = model.predict_proba(input_vector)[0]              # e.g., [0.2, 0.3, 0.5]
    prob_bert = predict_bert_prob(input_text, bert_model, bert_tokenizer)  # e.g., [0.1, 0.6, 0.3]

    prob_bert = np.array(prob_bert)
    avg_prob = (prob_lr + prob_bert) / 2

    prediction_index = np.argmax(avg_prob)
    labels = ['negative', 'neutral', 'positive']  # Match this with your model training
    return labels[prediction_index]


