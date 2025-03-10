import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForSeq2SeqLM
from keybert import KeyBERT
import spacy

# Load Named Entity Recognition (NER) Model
ner_model = "blaze999/Medical-NER"
tokenizer = AutoTokenizer.from_pretrained(ner_model)
model = AutoModelForTokenClassification.from_pretrained(ner_model)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load Summarization Model
summarization_model = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=summarization_model)

# Load Keyword Extraction Model
kw_model = KeyBERT()

# Load spaCy for text processing
nlp = spacy.load("en_core_web_sm")

# --- 1️⃣ Load the Transcript from a Text File ---
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# --- 2️⃣ Extract Named Entities (NER) ---
def extract_medical_entities(text):
    ner_results = ner_pipeline(text)

    medical_entities = {
        "Symptoms": [],
        "Treatment": [],
        "Diagnosis": [],
        "Prognosis": []
    }

    # Entity Mapping with Correction
    entity_mapping = {
        "SIGN_SYMPTOM": "Symptoms",
        "DISEASE_DISORDER": "Diagnosis",
        "THERAPEUTIC_PROCEDURE": "Treatment",
        "PROGNOSIS": "Prognosis",
        "MEDICATION": "Treatment"  # Fix misclassification of physiotherapy as medication
    }

    treatment_keywords = {"physiotherapy", "therapy", "sessions", "rehabilitation"}

    for entity in ner_results:
        entity_type = entity_mapping.get(entity["entity_group"], None)
        entity_text = entity["word"].lower()

        # Correct misclassified treatment terms
        if entity["entity_group"] == "MEDICATION" and entity_text in treatment_keywords:
            entity_type = "Treatment"

        if entity_type:
            medical_entities[entity_type].append(entity["word"])

    # Remove duplicates
    for key in medical_entities:
        medical_entities[key] = list(set(medical_entities[key]))

    return medical_entities

# --- 3️⃣ Generate Summary ---
def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# --- 4️⃣ Extract Keywords ---
def extract_keywords(text):
    doc = nlp(text)
    clean_text = " ".join([token.text for token in doc if not token.is_stop and token.is_alpha])
    keywords = kw_model.extract_keywords(clean_text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=10)
    return [kw[0] for kw in keywords]

# --- 🚀 Run the Full Pipeline ---
def generate_medical_report(file_path):
    transcript = load_text(file_path)

    medical_entities = extract_medical_entities(transcript)
    summary = summarize_text(transcript)
    keywords = extract_keywords(transcript)

    report = {
        "Patient_Name": "Unknown",
        "Symptoms": medical_entities["Symptoms"],
        "Diagnosis": medical_entities["Diagnosis"],
        "Treatment1": medical_entities["Treatment"],
        "Current_Status": "Unknown",
        "Prognosis": medical_entities["Prognosis"] if medical_entities["Prognosis"] else "Not mentioned",
        "Summary": summary,
        "Keywords": keywords
    }

    return report

# --- Example Usage ---
file_path = "transcript.txt"  # Replace with actual file path
medical_report = generate_medical_report(file_path)
print(medical_report)