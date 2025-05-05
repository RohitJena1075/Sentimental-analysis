# app.py
import streamlit as st
import numpy as np
from joblib import load
from model.model import predict_sentiment_ensemble
from model.bert_model import load_bert_model, predict_bert_prob

# Load models
model = load("model/model.joblib")
vectorizer = load("model/vect.joblib")
bert_model, tokenizer = load_bert_model()

# UI Config
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #0e1117; color: #fafafa;}
    .stTextArea textarea {font-size: 18px;}
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #ff4b4b, #ff914d);
        border: none;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
        This **Sentiment Analyzer** app uses an **ensemble of Machine Learning & BERT** to classify text as:
        - Positive 😊  
        - Neutral 😐  
        - Negative 😞

        Just type in a review or comment and click **Analyze** to see the result!
    """)

# Title Section
st.title("🧠 Sentiment Analyzer")
st.subheader("💬 Analyze the sentiment of any text — reviews, comments, or feedback!")

# Input Section
user_input = st.text_area("✍️ Enter your text below:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip():
        prediction = predict_sentiment_ensemble(model, vectorizer, bert_model, tokenizer, user_input)

        # Emoji-based feedback
        emojis = {"positive": "😊", "neutral": "😐", "negative": "😞"}
        st.success(f"Predicted Sentiment: **{prediction.capitalize()}** {emojis[prediction]}")

        # Show probability distribution
        prob_lr = model.predict_proba(vectorizer.transform([user_input]))[0]
        prob_bert = np.array(predict_bert_prob(user_input, bert_model, tokenizer))
        avg_prob = (prob_lr + prob_bert) / 2

        st.subheader("📊 Sentiment Probability")
        st.bar_chart({
            "Negative": avg_prob[0],
            "Neutral": avg_prob[1],
            "Positive": avg_prob[2]
        })

    else:
        st.warning("⚠️ Please enter some text to analyze.")

