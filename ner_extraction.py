
import torch
from transformers import pipeline
from keybert import KeyBERT
import spacy

# Load Named Entity Recognition (NER) Model from Hugging Face Hub
ner_pipeline = pipeline("ner", model="blaze999/Medical-NER", tokenizer="blaze999/Medical-NER", aggregation_strategy="simple")

# Load Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load Keyword Extraction Model
kw_model = KeyBERT()

# Load spaCy for text processing
nlp = spacy.load("en_core_web_sm")

# --- Load the Transcript from a Text File ---
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# --- Extract Named Entities (NER) ---
def extract_medical_entities(text):
    ner_results = ner_pipeline(text)
    medical_entities = {"Symptoms": [], "Treatment": [], "Diagnosis": [], "Prognosis": []}

    # Entity Mapping
    entity_mapping = {
        "SIGN_SYMPTOM": "Symptoms",
        "DISEASE_DISORDER": "Diagnosis",
        "THERAPEUTIC_PROCEDURE": "Treatment",
        "PROGNOSIS": "Prognosis",
        "MEDICATION": "Treatment"
    }

    # treatment_keywords = {"physiotherapy", "therapy", "sessions", "rehabilitation"}

    for entity in ner_results:
        entity_type = entity_mapping.get(entity["entity_group"], None)
        entity_text = entity["word"].lower()
        
        # Fix misclassification
        if entity["entity_group"] == "MEDICATION":
            entity_type = "Treatment"
        
        if entity_type:
            medical_entities[entity_type].append(entity["word"])

    # Remove duplicates
    for key in medical_entities:
        medical_entities[key] = list(set(medical_entities[key]))

    return medical_entities

# --- Generate Summary ---
def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# --- Extract Keywords ---
def extract_keywords(text):
    doc = nlp(text)
    clean_text = " ".join([token.text for token in doc if not token.is_stop and token.is_alpha])
    keywords = kw_model.extract_keywords(clean_text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=10)
    return [kw[0] for kw in keywords]

# --- Run Full Pipeline ---
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

# Example Usage
file_path = "transcript.txt"  # Replace with actual file path
medical_report = generate_medical_report(file_path)
print(medical_report)
