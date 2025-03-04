import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_path = "model"  # Directory where model files are stored
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("Sentiment Analysis using BERT")
st.write("Enter a review, and the model will classify it as positive or negative.")

# User input
user_input = st.text_area("Enter your review:")

if st.button("Classify"):
    if user_input:
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Convert logits to label
        sentiment = "Positive" if torch.argmax(outputs.logits).item() == 1 else "Negative"
        st.write(f"**Sentiment:** {sentiment}")
    else:
        st.write("⚠️ Please enter a review before classifying.")

