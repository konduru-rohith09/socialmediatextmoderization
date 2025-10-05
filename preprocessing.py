import os
import re
import pandas as pd
import nltk
import spacy
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

def load_csv_folder(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in all_files]
    if not dfs:
        raise ValueError("No CSV files found in folder")
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def text_preprocessing(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r"(.)\1{2,}", r'\1\1', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    return text.lower().strip()

def clean_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words]
    return " ".join(tokens)

def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = []
    for sent in sentences:
        words.extend(word_tokenize(sent))
    return " ".join([lemmatizer.lemmatize(w) for w in words])

def full_preprocessing(df, text_column='comment_text', method='nltk'):
    df['cleaned_text'] = df[text_column].apply(text_preprocessing)
    if method == 'spacy':
        df['lemmatized_text'] = df['cleaned_text'].apply(clean_text)
    else:
        df['lemmatized_text'] = df['cleaned_text'].apply(preprocess_text)
    return df
