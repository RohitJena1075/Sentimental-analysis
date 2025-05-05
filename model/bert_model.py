import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import os

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def train_bert_model(texts, labels, save_path='./bert_model'):
    if os.path.exists(os.path.join(save_path, 'pytorch_model.bin')):
        print(f"✅ Model already exists at {save_path}, skipping training.")
        return

    texts = texts[:2000]
    labels = labels[:2000]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    dataset = BERTDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_steps = 30
    step_count = 0
    total_loss = 0

    print(f"\n🔄 Starting training (max {max_steps} steps)...\n")
    for batch in dataloader:
        if step_count >= max_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"  Step {step_count}, Loss: {loss.item():.4f}")
        step_count += 1

    print(f"\n✅ Training completed. Avg Loss: {total_loss / step_count:.4f}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Model saved to: {save_path}")

def load_bert_model(path='./bert_model'):
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer

def predict_bert_prob(text, model, tokenizer):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

    return probs[0].tolist()  # returns list of class probabilities like [neg, neutral, pos]








