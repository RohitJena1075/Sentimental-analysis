import re
from nltk.corpus import stopwords

def preprocess_text(text):
    """
    Preprocesses a list of text data by removing special characters, converting to lowercase,
    tokenizing, and removing stopwords.
    """
    corpus = []
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')  # Keep "not" for sentiment analysis

    for i in text:
        review = re.sub('[^a-zA-Z]', ' ', i)  # Remove special characters and numbers
        review = review.lower()  # Convert to lowercase
        review = review.split()  # Tokenize
        review = [word for word in review if word not in set(all_stopwords)]  # Remove stopwords
        corpus.append(' '.join(review))  # Join words back into a single string

    return corpus