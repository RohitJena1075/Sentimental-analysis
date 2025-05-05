📊 Sentimental Analysis with Soft Voting Ensemble (Logistic Regression + BERT)
This project is a sentiment analysis system that uses a Soft Voting Ensemble of a traditional machine learning model (Logistic Regression) and a pretrained BERT transformer model to classify text as Positive or Negative. It also includes a Tkinter-based GUI for easy user interaction.

🚀 Features
✅ Combines Logistic Regression and BERT using soft voting for enhanced accuracy.
✅ Lightweight traditional ML component for faster inference.
✅ Powerful BERT component for deep semantic understanding.
✅ GUI built with Tkinter for user-friendly interaction.
✅ Easy to retrain, extend, or adapt for other domains.

📂 Project Structure
plaintext
Copy
Edit
Sentimental-analysis/
│
├── data/                   # Dataset folder
│   └── Twitter_Data.csv
│
├── model/                  # ML and BERT model-related files
│   ├── model.py
│   ├── model.joblib
│   ├── vect.joblib
│   └── train_bert_runner.py
│
├── gui/                    # GUI application
│   └── gui.py
│
├── bert_model/             # (Ignored in repo) Contains BERT weights
│
├── preprocessing.py        # Preprocessing utilities
├── sentiment_ana.py        # Main file to run GUI app
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignore rules
└── README.md               # Project documentation

🧠 Model Details
Traditional ML Model:
Algorithm: Logistic Regression
Trained on: TF-IDF features of cleaned tweets
Tools: scikit-learn
Transformer Model:
Base Model: bert-base-uncased (HuggingFace)
Fine-tuned on: 2000(case taken can be increased and train model) tweets from the dataset
Output: Show statement quality
Tools: transformers, torch
Ensemble Strategy:
Soft Voting: Weighted average of probability outputs from both models.
Final Decision: The class with the highest average probability.

💻 Installation and Setup
🔧 Prerequisites
Python 3.8+
pip
Git

🛠️ Clone the Repo
bash
Copy
Edit
git clone https://github.com/RohitJena1075/Sentimental-analysis.git
cd Sentimental-analysis

📦 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt

📥 Download BERT Model Weights (Optional)
If not included, run the fine-tuning or modify the bert_model.py to automatically download bert-base-uncased using HuggingFace:

python
Copy
Edit
from transformers import BertTokenizer, BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

🏁 How to Run
🖥️ Launch the GUI
bash
Copy
Edit
python sentiment_ana.py
A GUI will open where you can type a sentence or tweet, and the system will classify it as Positive or Negative and Neutral.


👥 Collaboration
We welcome contributions!
📌 How to Contribute:
Fork the repo.
Create a new branch (git checkout -b feature-name).
Make changes and commit (git commit -m 'Add new feature').
Push to your fork (git push origin feature-name).
Open a Pull Request.


📄 License
This project is licensed under the MIT License.
swift
Copy
Edit
MIT License
Copyright (c) 2025 Rohit Jena
Permission is hereby granted, free of charge, to any person obtaining a copy...
For full license, refer to LICENSE file.

🙌 Acknowledgements
HuggingFace Transformers
scikit-learn
Tkinter GUI Toolkit


