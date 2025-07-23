import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "imdb-distilbert-finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")

# Text input for the user
user_input = st.text_area("Enter your movie review:", "")

if st.button("Analyze Review"):
    if user_input.strip():
        # Get sentiment prediction
        result = sentiment_pipeline(user_input)[0]
        label = result['label']
        score = result['score']

        # Display the result
        if label == "LABEL_1":
            st.success(f"Positive review with confidence: {score:.2f}")
        else:
            st.error(f"Negative review with confidence: {score:.2f}")
    else:
        st.warning("Please enter a review to analyze.")
