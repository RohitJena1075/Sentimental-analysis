import pandas as pd
from bert_model import train_bert_model, load_bert_model

# Load dataset
df = pd.read_csv("Twitter_Data.csv")
df = df.dropna(subset=["clean_text", "category"])

# Ensure labels are integers (0 or 1 and 3)
df["category"] = pd.to_numeric(df["category"], errors="coerce")
df = df[df["category"].isin([0, 1, 2])]

texts = df["clean_text"].tolist()
labels = df["category"].astype(int).tolist()

# Train and save model
train_bert_model(texts, labels)

# ✅ Confirm it's loaded correctly
model, tokenizer = load_bert_model()





