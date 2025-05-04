import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

def train_and_save_model(corpus, labels, progress_callback=None):
    """
    Trains a Logistic Regression model and saves it along with the vectorizer.
    """
    # Vectorize the text
    vect = CountVectorizer(max_features=1500)
    X = vect.fit_transform(corpus).toarray()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

    # Notify progress
    if progress_callback:
        progress_callback("start")

    # Train the model
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # Notify progress
    if progress_callback:
        progress_callback("stop")

    # Save the model and vectorizer
    joblib.dump(classifier, "model.joblib")
    joblib.dump(vect, "vect.joblib")

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return cm, accuracy

def load_model_and_vectorizer():
    """
    Loads the saved model and vectorizer from joblib files.
    """
    try:
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vect.joblib")
        return model, vectorizer
    except FileNotFoundError as e:
        raise FileNotFoundError("Model or vectorizer file not found. Please ensure 'model.joblib' and 'vect.joblib' exist.") from e

def predict_sentiment(model, vectorizer, input_text):
    """
    Predicts the sentiment of the given input text using the loaded model and vectorizer.
    """
    # Preprocess the input text (basic preprocessing)
    input_vector = vectorizer.transform([input_text])  # Transform the input text into vectorized form
    prediction = model.predict(input_vector)  # Predict the sentiment
    return prediction[0]  # Return the predicted label