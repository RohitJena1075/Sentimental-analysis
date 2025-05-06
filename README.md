# 🧠 Sentiment Analysis 

A Python-based Sentiment Analysis project combining **BERT** and **Logistic Regression** in a **Soft Voting Ensemble**, featuring both a **Tkinter GUI** and a **Streamlit Web App**. Designed to classify textual sentiments from Twitter data into **Negative**, **Neutral**, or **Positive**.

---

## 📁 Project Structure

```
Sentimental-analysis/
│
├── app.py # Streamlit web interface
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
├── sentiment_ana.py # Entry point for GUI
├── requirements.txt
└── README.md
```

- yaml
- Copy
- Edit


---

## 🚀 Features

- ✔️ Soft Voting Ensemble: BERT + Logistic Regression  
- ✔️ Multi-class Sentiment Classification: Negative, Neutral, Positive  
- ✔️ Two Interfaces:
  - 🖥️ Tkinter-based Desktop GUI  
  - 🌐 Streamlit-based Web App  
- ✔️ Pretrained models saved with `joblib`  
- ✔️ Real-time predictions from user input  
- ✔️ Lightweight classical ML + deep contextual language model  

---

## 🤖 Model Details

### 1. Logistic Regression (Baseline)
- TF-IDF vectorization of text
- Efficient and fast for text classification tasks

### 2. BERT Model
- Based on Hugging Face’s `bert-base-uncased`
- Fine-tuned on 2000 Twitter samples (increased accordingly)
- user input dataset trained model can be made
- Generates deep contextual embeddings
- Returns logits and probabilities for sentiment classes

### 3. Soft Voting Ensemble
- Combines output probabilities of BERT and Logistic Regression
- Final prediction = `argmax(weighted_average(probabilities))`

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

- git clone https://github.com/RohitJena1075/Sentimental-analysis.git
- cd Sentimental-analysis
---

### 2. Setup Environment

- Recommended: Use a virtual environment.
- bash
- Copy
- Edit
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate
- pip install -r requirements.txt

---

### 3. Run the Application

- bash
- Copy
- Edit
- python sentiment_ana.py
- A GUI window will appear for sentiment input.

---

### 🌐 Web Deployment (Hugging Face Spaces or Streamlit Sharing)
To deploy online:
- Track large model files using Git LFS:

- bash
- Copy
- Edit
- git lfs install
- git lfs track "bert_model/*"
- git lfs track "model/*.joblib"
- Push to Hugging Face or deploy using Streamlit Cloud.

---

### 📁 .gitignore Highlights

# Byte-compiled / optimized / DLL files
- __pycache__/
- *.pyc

# Virtual environments
- .venv/
- venv/
- env/

# Large model files
- bert_model/
- *.joblib
- *.bin
- *.pt
- *.safetensors
- *.zip

# Jupyter & system files
- .ipynb_checkpoints/
- .DS_Store
- Thumbs.db

# Temp/log files
- *.log
- *.tmp

---

### 📜 License
This project is licensed under the MIT License. See LICENSE for more information.

---

### 🤝 Contributing
- Pull requests are welcome! If you’d like to:
- Report bugs
- Suggest features
- Improve documentation or code
- Please fork the repo and submit a PR.

---

### 👤 Author
- Rohit Jena
- GitHub (https://github.com/RohitJena1075)

---

### 🙌 Acknowledgements
- Hugging Face Transformers
- Scikit-learn
- Tkinter GUI
- Twitter Data contributors


