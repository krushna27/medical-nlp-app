import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, pipeline
import os
from sklearn.metrics import accuracy_score, f1_score

# --- 1. Data Preparation ---

# --- Sentiment Data ---
sentiment_data = {'text': [
    "I'm really worried about these symptoms.",
    "The doctor explained everything clearly, feeling much better now.",
    "Just a routine check-up, no concerns.",
    "I feel so anxious waiting for my test results.",
    "Finally got some good news from the clinic!",
    "Everything seems normal, which is reassuring.",
    "This pain is unbearable, I'm scared.",
    "The nurse was very kind and understanding, calmed me down.",
    "No issues detected, feeling relieved.",
    "I'm panicking about my upcoming surgery."
],
'sentiment': ['Anxious', 'Reassured', 'Neutral', 'Anxious', 'Reassured', 'Neutral', 'Anxious', 'Reassured', 'Neutral', 'Anxious']}

sentiment_df = pd.DataFrame(sentiment_data)

# --- Intent Data ---
intent_data = {'text': [
    "I'm having chest pains, could this be serious?",
    "Just wanted to update you that I'm feeling a bit better today.",
    "Is it normal to feel this tired after the medication?",
    "I need to reschedule my appointment.",
    "Thank you for your help, I appreciate it.",
    "I'm still coughing and feeling weak.",
    "What are the possible side effects of this drug?",
    "Could you please send me my medical records?",
    "I'm experiencing new symptoms since yesterday.",
    "Just wanted to let you know I'm recovering well."
],
'intent': ['Reporting symptoms', 'Providing updates', 'Asking questions', 'Scheduling appointment', 'Thanking/Appreciating', 'Reporting symptoms', 'Asking questions', 'Requesting records', 'Reporting symptoms', 'Providing updates']}

intent_df = pd.DataFrame(intent_data)

# --- Define Output Directories ---
sentiment_model_path = "./fine_tuned_sentiment_model"
intent_model_path = "./fine_tuned_intent_model"

for path in [sentiment_model_path, intent_model_path]:
    os.makedirs(path, exist_ok=True)

# --- 2. Sentiment Model Training ---

# a) Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_texts_sentiment = sentiment_df['text'].tolist()
train_labels_sentiment_str = sentiment_df['sentiment'].tolist()

train_encodings_sentiment = tokenizer(train_texts_sentiment, truncation=True, padding=True)

# b) Label Encoding
label_encoder_sentiment = LabelEncoder()
train_labels_sentiment = label_encoder_sentiment.fit_transform(train_labels_sentiment_str)
num_labels_sentiment = len(label_encoder_sentiment.classes_)

# c) Dataset Class
class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset_sentiment = PatientDataset(train_encodings_sentiment, train_labels_sentiment)

# d) Model Initialization
model_sentiment = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels_sentiment)

# e) Training Configuration
training_args_sentiment = TrainingArguments(
    output_dir=sentiment_model_path,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs"
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(labels, preds), 'f1_score': f1_score(labels, preds, average="weighted")}

# f) Trainer & Training
trainer_sentiment = Trainer(
    model=model_sentiment,
    args=training_args_sentiment,
    train_dataset=train_dataset_sentiment,
    eval_dataset=train_dataset_sentiment,
    compute_metrics=compute_metrics,
)

print("--- Training Sentiment Model ---")
trainer_sentiment.train()
trainer_sentiment.save_model(sentiment_model_path)
tokenizer.save_pretrained(sentiment_model_path)

# --- 3. Intent Model Training ---
train_texts_intent = intent_df['text'].tolist()
train_labels_intent_str = intent_df['intent'].tolist()
train_encodings_intent = tokenizer(train_texts_intent, truncation=True, padding=True)

# a) Label Encoding
label_encoder_intent = LabelEncoder()
train_labels_intent = label_encoder_intent.fit_transform(train_labels_intent_str)
num_labels_intent = len(label_encoder_intent.classes_)

train_dataset_intent = PatientDataset(train_encodings_intent, train_labels_intent)

# b) Model Initialization
model_intent = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels_intent)

# c) Training Configuration
training_args_intent = TrainingArguments(
    output_dir=intent_model_path,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs"
)

trainer_intent = Trainer(
    model=model_intent,
    args=training_args_intent,
    train_dataset=train_dataset_intent,
    eval_dataset=train_dataset_intent,
    compute_metrics=compute_metrics,
)

print("--- Training Intent Model ---")
trainer_intent.train()
trainer_intent.save_model(intent_model_path)
tokenizer.save_pretrained(intent_model_path)

# --- 4. Inference Function ---
def analyze_patient_text(text):
    """Analyzes patient text for sentiment and intent."""
    sentiment_pipeline = pipeline("text-classification", model=sentiment_model_path, tokenizer=sentiment_model_path)
    intent_pipeline = pipeline("text-classification", model=intent_model_path, tokenizer=intent_model_path)

    sentiment_result = sentiment_pipeline(text)[0]
    intent_result = intent_pipeline(text)[0]

    sentiment_index = int(sentiment_result['label'].split('_')[-1])
    intent_index = int(intent_result['label'].split('_')[-1])

    predicted_sentiment = label_encoder_sentiment.classes_[sentiment_index]
    predicted_intent = label_encoder_intent.classes_[intent_index]

    return {
        "Sentiment": predicted_sentiment,
        "Intent": predicted_intent,
        "Sentiment_Confidence": sentiment_result['score'],
        "Intent_Confidence": intent_result['score']
    }

# --- 5. Example Usage ---
patient_message = "I'm feeling very uneasy and my heart is racing. I'm not sure what to do."
analysis_result = analyze_patient_text(patient_message)
print("\n--- Analysis Result ---")
print(f"Patient Message: '{patient_message}'")
print("Analysis:", analysis_result)
