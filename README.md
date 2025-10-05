# 🧠 SOCIAL MEDIA CONTEXT MODERIZATION

A **BERT-powered AI system** that automatically detects and filters **toxic, offensive, or inappropriate text** from social media posts and comments.  
This project applies **Natural Language Processing (NLP)** and **Machine Learning (ML)** to ensure respectful online communication.

---

## 🎯 Objective

The goal of this project is to **analyze user-generated text** and predict whether it contains **toxic or harmful language** (like hate speech, insults, or threats).  
It helps moderators and platforms automatically flag or remove inappropriate content.

---

## 🚀 Features

- ✅ Detects multiple categories of toxicity — *toxic, severe toxic, obscene, threat, insult, identity hate, etc.*
- ⚙️ Uses **BERT (Bidirectional Encoder Representations from Transformers)** for deep text understanding
- 🧹 Includes **advanced text preprocessing** — tokenization, lemmatization, stopword & emoji handling
- 🌐 Has a **FastAPI backend** and a **simple HTML frontend** for user interaction
- 📦 Easy to deploy locally or on cloud platforms like **AWS**, **GCP**, or **Hugging Face Spaces**
- 🧠 Includes a **Jupyter Notebook** for training and fine-tuning models

  ---

## 🧩 Project Structure

socialmediatextmoderization/
│
├── Backend/
│ ├── main.py
│ ├── toxic_bert_model/
│ │ ├── config.json
│ │ ├── tokenizer.json
│ │ └── pytorch_model.bin
│ ├── utils/
│ │ └── preprocessing.py
│ └── models/
│ └── text_model.py
│
├── frontend/
│ └── index.html
│
├── data/
│ └── text/
│ ├── combined_dataset.csv
│ └── other_dataset_files
│
├── text_combine_csv.py
└── text.ipynb


---

## 🧠 `Backend/main.py`

This is the **main FastAPI application** that connects the model with the user interface.

### 📝 What it does:
- Loads the **BERT model** and **tokenizer** from `toxic_bert_model/`
- Accepts input text from the frontend (via `POST /predict`)
- Cleans and preprocesses the text using `utils/preprocessing.py`
- Passes the cleaned text to the model (`models/text_model.py`)
- Returns the prediction result as JSON

### 💻 Example Code Snippet:

@app.post("/predict")
def predict_text(data: TextInput):
    cleaned_text = preprocess_text(data.text)
    prediction = model.predict(cleaned_text)
    return {"final_label": prediction}


⚙️ How it works (Step-by-Step):

User enters text in the web form.
The text is sent to the backend via the /predict endpoint.
The backend cleans the text → sends it to the BERT model.
Model predicts toxicity category (e.g., “toxic”, “clean”).
The result is returned to the frontend for display.

🧰 Backend/utils/preprocessing.py

This module handles text cleaning and preparation before the data is sent to the model.

🧹 Operations Performed:

Converts text to lowercase
Removes URLs, HTML tags, emojis, and punctuation
Tokenizes sentences into words
Removes stopwords
Lemmatizes words to their base form

💻 Example Code:
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)     # Remove URLs
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords]
    return " ".join(tokens)

🎯 Purpose:

Makes input text clean and consistent, improving the accuracy of the BERT model by reducing noise and irrelevant patterns.

🧩 Backend/models/text_model.py

Defines and loads the BERT model used for prediction.

⚙️ What Happens Here:

Loads the pretrained BERT model and tokenizer
Converts text into token IDs understandable by BERT
Passes data through the model to get prediction probabilities
Maps model output to human-readable labels (like “toxic”)

💻 Example Logic:
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = torch.sigmoid(outputs.logits)

🔄 Working:

Text → Tokenized into numerical IDs

BERT → Generates contextual embeddings

Classification layer → Predicts toxicity probabilities

💻 frontend/index.html

A simple web interface that allows users to enter text and see the toxicity result.

🌐 Features:

Input box for text
“Analyze” button to submit
Displays model’s toxicity prediction dynamically

💻 Example Snippet:
<form id="predict-form">
  <textarea id="user-input" placeholder="Enter text here..."></textarea>
  <button type="submit">Analyze</button>
</form>
<div id="result"></div>

⚙️ Working:

When the user clicks “Analyze,” the frontend sends the text to /predict API.
The backend processes it and returns the result (e.g., Toxic / Not Toxic).

📊 data/text/combined_dataset.csv

Contains the training dataset used for fine-tuning the model.
Each row includes text and corresponding labels for toxicity classes.

Example columns:

text, toxic, severe_toxic, obscene, threat, insult, identity_hate
Used by text.ipynb during training.

🧮 text_combine_csv.py

A simple Python script that merges multiple CSV datasets into one large dataset for model training.

💻 Example Logic:
import pandas as pd

files = ["toxic.csv", "comments.csv"]
combined = pd.concat([pd.read_csv(f) for f in files])
combined.to_csv("data/text/combined_dataset.csv", index=False)

🎯 Purpose:

Helps combine datasets from different sources efficiently into one file.

📓 text.ipynb

Jupyter Notebook used to train and fine-tune the BERT model.

🧠 Steps Performed:

Load dataset (combined_dataset.csv)
Preprocess text data
Tokenize text using BERT tokenizer
Train classification layers
Save trained model files into Backend/toxic_bert_model/

💻 Example Snippet:
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/konduru-rohith09/socialmediatextmoderization.git
cd socialmediatextmoderization

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt


If requirements.txt is missing:

pip install fastapi uvicorn torch transformers pandas numpy scikit-learn spacy nltk emoji
python -m spacy download en_core_web_sm

▶️ Running the Application
🖥️ Backend (FastAPI)
cd Backend
uvicorn main:app --reload


➡ Server runs at: http://127.0.0.1:8000

🌐 Frontend

Open your browser and visit the above URL to use the text moderation interface.

🧰 Troubleshooting
Issue	Solution
Model not found	Ensure Backend/toxic_bert_model/ contains files like pytorch_model.bin
TemplateNotFound: index.html	Place index.html inside the frontend/ folder
nltk data missing	Run:
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

🌟 Future Enhancements

🌍 Multilingual text moderation
📊 Real-time toxicity analytics dashboard
🚀 Docker / Streamlit / Hugging Face Spaces deployment
💡 Explainable AI (highlight toxic words in sentences)

🤝 Contributing

Fork this repository

Create a feature branch:
  git checkout -b feature-name

Commit changes:
  git commit -m "Added new feature"


Push to your branch:
  git push origin feature-name

Open a Pull Request 🚀

📜 License

Licensed under the MIT License.
See the LICENSE file for more information.

👨‍💻 Author

Konduru Rohith
📎 GitHub: @konduru-rohith09

💬 Acknowledgements

🤗 Hugging Face Transformers
⚡ FastAPI Documentation
🧠 spaCy NLP Toolkit
📚 NLTK Python Library





