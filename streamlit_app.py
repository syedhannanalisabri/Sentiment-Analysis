import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle
import os

# App configuration
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")

# Constants
max_words = 5000
max_len = 100
embedding_dim = 128
num_classes = 3

@st.cache_resource
def load_model_and_tokenizer():
    # Load the pre-trained model
    try:
        model = tf.keras.models.load_model('sentiment_model.h5')
    except:
        st.error("Model file not found. Please ensure 'sentiment_model.h5' is in the app directory.")
        return None, None
    
    # Load the tokenizer
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except:
        st.error("Tokenizer file not found. Please ensure 'tokenizer.pickle' is in the app directory.")
        return None, None
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    if not model or not tokenizer:
        return None, None
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    pred = model.predict(padded, verbose=0)[0]
    class_idx = np.argmax(pred)
    
    # Map index to sentiment
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_map[class_idx]
    
    return sentiment, pred[class_idx]

# Main app
def main():
    st.title("🐦 Twitter Sentiment Analysis")
    st.markdown("""
    Enter a tweet below to predict its sentiment (positive, negative, or neutral).
    The model was trained on Twitter data using an LSTM neural network.
    """)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model and tokenizer:
        # Input form
        with st.form(key='tweet_form'):
            tweet = st.text_area("Enter your tweet (max 280 characters):", max_chars=280, height=100)
            submit_button = st.form_submit_button("Analyze Sentiment")
            
            if submit_button and tweet.strip():
                sentiment, prob = predict_sentiment(tweet, model, tokenizer)
                
                if sentiment:
                    # Display results with emoji
                    sentiment_emoji = {
                        'positive': '😊👍',
                        'negative': '😔👎',
                        'neutral': '😐'
                    }
                    
                    st.success(f"**Predicted Sentiment**: {sentiment.capitalize()} {sentiment_emoji[sentiment]}")
                    st.info(f"**Confidence**: {prob:.4f}")
                    st.write(f"**Tweet**: {tweet}")
                else:
                    st.error("Error processing the tweet. Please try again.")
            elif submit_button and not tweet.strip():
                st.warning("Please enter a tweet to analyze.")
    
    # Add some styling
    st.markdown("""
    <style>
    .stTextArea textarea {
        font-family: Arial, sans-serif;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
