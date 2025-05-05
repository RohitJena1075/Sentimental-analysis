# app.py
import streamlit as st
from model import predict_sentiment_ensemble
from joblib import load

# Load models
model = load("model/model.joblib")
vectorizer = load("model/vect.joblib")
# Assuming you have a load_bert_model function
from bert_model import load_bert_model

bert_model, tokenizer = load_bert_model()

# App title
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analyzer")
st.subheader("Enter a review or comment below:")

# User input
user_input = st.text_area("Your Text Here", height=150)

if st.button("Analyze"):
    if user_input.strip():
        prediction = predict_sentiment_ensemble(model, vectorizer, bert_model, tokenizer, user_input)
        st.success(f"Predicted Sentiment: **{prediction.capitalize()}**")

    else:
        st.warning("Please enter some text for analysis.")
