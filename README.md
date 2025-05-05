# 🧠 Sentiment Analysis App

A Python-based Sentiment Analysis project combining **BERT** and **Logistic Regression** in a **Soft Voting Ensemble** with a Tkinter GUI. Designed to classify textual sentiments from Twitter data into **Negative**, **Neutral**, or **Positive**.

---

## 📂 Project Structure

Sentimental-analysis/
│
├── data/
│ └── Twitter_Data.csv # Dataset used
│
├── gui/
│ └── gui.py # Tkinter GUI code
│
├── model/
│ ├── model.py # ML + ensemble logic
│ ├── bert_model.py # BERT model loading
│ ├── preprocessing.py # Text preprocessing
│ ├── train_bert_runner.py # BERT training script
│ ├── model.joblib # Saved Logistic Regression model
│ └── vect.joblib # Saved TF-IDF vectorizer
│
├── .gitignore
├── sentiment_ana.py # Entry point to launch GUI
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🚀 Features

- ✔️ Soft Voting Ensemble of BERT + Logistic Regression
- ✔️ Multi-class Sentiment: Negative, Neutral, Positive
- ✔️ User-friendly GUI with `tkinter`
- ✔️ Lightweight classical ML model + deep contextual language model
- ✔️ Easy model saving and reuse (`joblib`)
- ✔️ Real-time predictions from user input

---

## 🤖 Model Details

### 1. Logistic Regression (Baseline)
- Uses TF-IDF vectorization
- Fast and efficient for baseline sentiment classification

### 2. BERT Model
- Pre-trained on large corpora
- Fine-tuned on 2000 Twitter samples
- Hugging Face `bert-base-uncased` backbone
- Output: Sentiment logits and probabilities

### 3. Soft Voting Ensemble
- Combines probabilities of both models
- Output = `argmax(weighted_average(probs))`

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/Sentimental-analysis.git
cd Sentimental-analysis

---

### 2. Setup Environment

Recommended: Use a virtual environment.
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

---

### 3. Run the Application

bash
Copy
Edit
python sentiment_ana.py
A GUI window will appear for sentiment input.

---

📁 .gitignore Highlights
markdown
Copy
Edit
# Ignored files/folders
*.joblib
*.pth
__pycache__/
.venv/
*.bin
*.pt
bert_model/
📜 License
This project is licensed under the MIT License. See LICENSE for more information.

🤝 Contributing
Pull requests are welcome! If you’d like to:

Report bugs

Suggest features

Improve documentation or code

Please fork the repo and submit a PR.

👤 Author
Rohit Jena
GitHub

🙌 Acknowledgements
Hugging Face Transformers
Scikit-learn
Tkinter GUI
Twitter Data contributors

