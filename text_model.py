import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def compute_metrics(eval_pred, threshold=0.3):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))

    # 🔹 Support for per-label thresholds
    if isinstance(threshold, (float, int)):
        preds = (probs >= threshold).astype(int)
    else:  # list/array of thresholds per label
        threshold_arr = np.array(threshold).reshape(1, -1)
        preds = (probs >= threshold_arr).astype(int)

    labels = labels.astype(int)
    return {
        "accuracy": (preds == labels).mean(),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
        "f1": f1_score(labels, preds, average="micro", zero_division=0)
    }

def train_bert_model(x_train, y_train, x_val, y_val,
                     model_name="bert-base-uncased",
                     output_dir="./results",
                     epochs=3, batch_size=8,
                     max_length=128, threshold=0.3):

    num_labels = y_train.shape[1]

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(list(x_train), truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(list(x_val), truncation=True, padding=True, max_length=max_length)

    train_dataset = ToxicDataset(train_encodings, y_train.to_numpy())
    eval_dataset = ToxicDataset(val_encodings, y_val.to_numpy())

    # Compute pos_weight for BCEWithLogitsLoss
    pos_counts = y_train.sum(axis=0).values
    total_counts = y_train.shape[0]
    pos_weight = torch.tensor((total_counts - pos_counts) / (pos_counts + 1e-5), dtype=torch.float)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=20,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, threshold=threshold)
    )

    trainer.train()
    results = trainer.evaluate()
    return model, tokenizer, trainer, results

def predict_texts(model, tokenizer, texts, max_length=128, threshold=0.3):
    encodings = tokenizer(list(texts), truncation=True, padding=True,
                          max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.sigmoid(outputs.logits)

    # 🔹 Support for per-label thresholds
    if isinstance(threshold, (float, int)):
        preds = (probs >= threshold).int()
    else:  # list/array of thresholds per label
        threshold_tensor = torch.tensor(threshold).unsqueeze(0)  # shape: (1, num_labels)
        preds = (probs >= threshold_tensor).int()

    return preds, probs
