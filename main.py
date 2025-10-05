import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from fastapi import FastAPI, Request, Form

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils.preprocessing import text_preprocessing, clean_text, preprocess_text

## labels and threshold
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult"]
threshold = 0.3

# Paths(where my files are located)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "toxic_bert_model")
TEMPLATES_DIR = os.path.join(BASE_DIR, "../frontend")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

# load tokenizer model
num_labels = 5
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=6, local_files_only=True)
model.eval()

# fastapi templates
app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)

#routes
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request, text: str = Form(...)):
    # Preprocess text
    clean = text_preprocessing(text)
    clean = preprocess_text(clean)

    # Tokenize input
    encodings = tokenizer([clean], truncation=True, padding=True, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.sigmoid(outputs.logits).squeeze()
        preds = (probs >= threshold).int()

    # Format results
    results = {label: {"prediction": int(preds[i].item()), "probability": float(probs[i].item())} 
               for i, label in enumerate(label_cols)}

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": results, "input_text": text}
    )



#run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
