import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from model import load_model_and_vectorizer, predict_sentiment

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis")
        self.root.geometry("500x800")
        self.root.configure(bg="black")
        self.model = None
        self.vectorizer = None
        self.progress = None
        self.setup_ui()

    def clear(self):
        entry1.delete(0, END)

    def exit(self):
        self.root.destroy()

    def predict(self):
        review = entry1.get().strip()
        if not review or len(review) < 5:
            messagebox.showerror("Error", "Please enter a meaningful review (at least 5 characters).")
            return

        try:
            # Load the model and vectorizer if not already loaded
            if self.model is None or self.vectorizer is None:
                self.model, self.vectorizer = load_model_and_vectorizer()

            # Predict sentiment
            sentiment = predict_sentiment(self.model, self.vectorizer, review)
            messagebox.showinfo("Result", f"Predicted Sentiment: {sentiment}")
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))

    def setup_ui(self):
        global entry1
        label1 = Label(self.root, text="Sentiment Analysis", font=("Arial", 20, "bold"), bg="black", fg="white")
        label1.pack(pady=10)
        label2 = Label(self.root, text="Enter a sample", font=("Arial", 15, "bold"), bg="black", fg="white")
        label2.pack(pady=10)
        entry1 = Entry(self.root, width=50)
        entry1.pack(pady=10)
        button1 = Button(self.root, text="Predict", font=("Arial", 15, "bold"), bg="black", fg="white", command=self.predict)
        button1.pack(pady=10)
        button2 = Button(self.root, text="Clear", font=("Arial", 15, "bold"), bg="black", fg="white", command=self.clear)
        button2.pack(pady=10)
        button3 = Button(self.root, text="Exit", font=("Arial", 15, "bold"), bg="black", fg="white", command=self.exit)
        button3.pack(pady=10)

        # Add progress bar
        self.progress = ttk.Progressbar(self.root, orient=HORIZONTAL, length=300, mode='indeterminate')
        self.progress.pack(pady=10)