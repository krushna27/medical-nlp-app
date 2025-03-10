import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from keybert import KeyBERT
import spacy

# Load Models
@st.cache_resource
def load_models():
    ner_model = "blaze999/Medical-NER"
    tokenizer_ner = AutoTokenizer.from_pretrained(ner_model)
    model_ner = AutoModelForTokenClassification.from_pretrained(ner_model)
    ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner, aggregation_strategy="simple")

    summarization_model = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=summarization_model)

    kw_model = KeyBERT()

    return ner_pipeline, summarizer, kw_model

ner_pipeline, summarizer, kw_model = load_models()

st.title("🏥 Medical NLP Analysis")
user_text = st.text_area("Enter Patient Text", "")

if st.button("Analyze"):
    if user_text:
        ner_results = ner_pipeline(user_text)
        summary = summarizer(user_text, max_length=100, min_length=30, do_sample=False)
        keywords = kw_model.extract_keywords(user_text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=10)

        st.subheader("🧬 Named Entities")
        for entity in ner_results:
            st.write(f"- **{entity['word']}** → {entity['entity_group']}")

        st.subheader("📄 Summary")
        st.write(summary[0]["summary_text"])

        st.subheader("🔑 Keywords")
        st.write([kw[0] for kw in keywords])
    else:
        st.warning("⚠️ Please enter some text.")
