import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from nltk import corpus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import joblib
import os
import warnings
nltk.download('corpus')
nltk.download('stopwords')
warnings.filterwarnings("ignore")
vect =0
classifier=0
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("500x800")
root.configure(bg="black")
def clear():
    entry1.delete(0,END)
def exit():
    root.destroy()
def predict():
    if entry1.get() == "":
        messagebox.showerror("Error","Please enter a review")
    else:
        review = entry1.get()
        review = re.sub('[^a-zA-Z]',' ',review)
        review = review.split()
        review = [r.lower() for r in review]
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        X = vect.transform([review])
        y_pred = classifier.predict(X)
        if y_pred == "Positive":
            messagebox.showinfo("Result","Positive")
        elif y_pred == "Neutral":
            messagebox.showinfo("Result","Neutral")
        elif y_pred == "Negative":
            messagebox.showinfo("Result","Negative")
        else: 
            messagebox.showinfo("Result","Exception")
def wordcloud():
    df = pd.read_csv("Twitter_Data.csv")
    text= df['clean_text'].head(5000)
    text= text.dropna()
    corpus = ""
    for i in text:
        review = re.sub('[^a-zA-Z]',' ',i)
        review = review.lower()
        review = review.split()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [word for word in review if not word in set(all_stopwords)]
        corpus = ' '.join(review) 
    wordcloud = WordCloud(width=800,height=800,background_color='white',min_font_size=2).generate(corpus)
    plt.figure(figsize=(8,8),facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
def accuracy():
    df = pd.read_csv("Twitter_Data.csv")
    corpus = []
    for i in range(0,1000):
        review = re.sub('[^a-zA-Z]',' ',df['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    vect = CountVectorizer(max_features=1500)
    X = vect.fit_transform(corpus).toarray()
    y = df.iloc[:,1].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix: ",cm)
    print("Accuracy: ",accuracy_score(y_test,y_pred))
def save():
    df = pd.read_csv("Twitter_Data.csv")
    text= df['clean_text'].head(5000)
    text= text.dropna()
    corpus = ""
    for i in text:
        review = re.sub('[^a-zA-Z]',' ',i)
        review = review.lower()
        review = review.split()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [word for word in review if not word in set(all_stopwords)]
        corpus.append(review)
    vect = CountVectorizer(max_features=1500)
    X = vect.fit_transform(corpus).toarray()
    y = df.iloc[:,1].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    joblib.dump(classifier,"model.joblib")
    joblib.dump(vect,"vect.pkl")
    messagebox.showinfo("Result","Model Saved Successfully")
def load():
    if os.path.exists("model.joblib"):
        global classifier 
        classifier= joblib.load("model.joblib")
        global vect 
        vect = joblib.load("vect.joblib")
        messagebox.showinfo("Result","Model Loaded Successfully")
    else:
        messagebox.showerror("Error","Model not found")
def about():
    messagebox.showinfo("About","This is a Sentiment Analysis project using Logistic Regression")
def help():
    messagebox.showinfo("Help","Enter a review and click on Predict button to predict whether the review is positive or negative")


label1 = Label(root,text="Sentiment Analysis",font=("Arial",20,"bold"),bg="black",fg="white")
label1.pack(pady=10)
label2 = Label(root,text="Enter a sample",font=("Arial",15,"bold"),bg="black",fg="white")
label2.pack(pady=10)
entry1 = Entry(root,width=50)
entry1.pack(pady=10)
button1 = Button(root,text="Predict",font=("Arial",15,"bold"),bg="black",fg="white",command=predict)
button1.pack(pady=10)
button2 = Button(root,text="Clear",font=("Arial",15,"bold"),bg="black",fg="white",command=clear)
button2.pack(pady=10)
button3 = Button(root,text="Exit",font=("Arial",15,"bold"),bg="black",fg="white",command=exit)
button3.pack(pady=10)
button4 = Button(root,text="Wordcloud",font=("Arial",15,"bold"),bg="black",fg="white",command=wordcloud)
button4.pack(pady=10)
button5 = Button(root,text="Accuracy",font=("Arial",15,"bold"),bg="black",fg="white",command=accuracy)
button5.pack(pady=10)
button6 = Button(root,text="Save",font=("Arial",15,"bold"),bg="black",fg="white",command=save)
button6.pack(pady=10)
button7 = Button(root,text="Load",font=("Arial",15,"bold"),bg="black",fg="white",command=load)
button7.pack(pady=10)
button8 = Button(root,text="About",font=("Arial",15,"bold"),bg="black",fg="white",command=about)
button8.pack(pady=10)
button9 = Button(root,text="Help",font=("Arial",15,"bold"),bg="black",fg="white",command=help)
button9.pack(pady=10)
root.mainloop()