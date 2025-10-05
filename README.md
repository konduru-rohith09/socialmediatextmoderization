# 🧠 Social Media Text Moderization

A **BERT-powered social media text moderation system** that automatically detects and filters toxic, offensive, or inappropriate text content.  
This project leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** models to ensure a safer and more respectful online environment.

---

## 🚀 Features

- 🔍 Detects multiple toxicity categories — *toxic, severe toxic, obscene, threat, insult, identity hate, etc.*
- ⚙️ Built using **BERT** and **FastAPI**
- 🧹 Advanced text preprocessing — tokenization, lemmatization, stopword removal, emoji handling
- 🌐 Interactive web interface built with **HTML + FastAPI templates**
- 📦 Easy to deploy locally or on the cloud
- 🧠 Supports model training and fine-tuning via Jupyter Notebook

---

## 🧩 Project Structure

socialmediatextmoderization/
│
├── Backend/
│ ├── main.py # FastAPI app to serve the model
│ ├── toxic_bert_model/ # Trained BERT model & tokenizer files
│ ├── utils/
│ │ └── preprocessing.py # Text preprocessing utilities
│ └── models/
│ └── text_model.py # Model architecture and inference logic
│
├── frontend/
│ └── index.html # Web interface for text moderation
│
├── data/
│ └── text/
│ ├── combined_dataset.csv # Training dataset
│ └── other_dataset_files
│
├── text_combine_csv.py # Script to merge multiple datasets
└── text.ipynb # Jupyter Notebook for model training


---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.x |
| **Frameworks** | FastAPI, Jinja2 |
| **ML / NLP Libraries** | Transformers (BERT), PyTorch, scikit-learn, spaCy, nltk |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Data Handling** | pandas, numpy |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/konduru-rohith09/socialmediatextmoderization.git
cd socialmediatextmoderization
2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate3️⃣ Install Dependencies
pip install -r requirements.txt


If no requirements.txt file is available, install manually:

pip install fastapi uvicorn torch transformers pandas numpy scikit-learn spacy nltk emoji
python -m spacy download en_core_web_sm

🧠 Model Training
To train or fine-tune the model, open and run the notebook:
jupyter notebook text.ipynb


This notebook will:
✅ Preprocess the dataset
✅ Train a BERT-based classifier
✅ Save the model and tokenizer to Backend/toxic_bert_model/

▶️ Running the Application
🔹 Backend (FastAPI Server)
cd Backend
uvicorn main:app --reload


Once started, the server will be available at:

http://127.0.0.1:8000

🔹 Frontend

Open your browser and visit the above URL to access the interactive text moderation interface.

🧰 Troubleshooting
Issue	Solution
Model directory not found	Ensure Backend/toxic_bert_model/ contains model files like pytorch_model.bin
TemplateNotFound: index.html	Place index.html inside the frontend/ folder
nltk data missing	Run:
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

🌟 Future Enhancements

🧩 Multilingual text moderation support
📊 Real-time dashboard for toxicity analytics
🚀 Deploy using Docker, Streamlit, or Hugging Face Spaces

🤝 Contributing
Contributions are welcome!
To contribute:
Fork this repository

Create a new feature branch (git checkout -b feature-name)
Commit your changes

Push to the branch (git push origin feature-name)
Open a Pull Request

📜 License
This project is licensed under the MIT License.
See the LICENSE
 file for details.

👨‍💻 Author
Konduru Rohith
📎 GitHub: @konduru-rohith09

💬 Acknowledgements
Hugging Face Transformers
FastAPI Documentation
spaCy NLP Toolkit
nltk Python Library



