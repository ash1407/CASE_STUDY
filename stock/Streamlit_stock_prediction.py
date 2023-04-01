import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# loading the saved models

tokenizer = Tokenizer()

# Load the tokenizer from a local file
with open("/app/case_study/stock/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f) 

# Load the model from a local file
with open("/app/case_study/stock/xg_reg_model.pickle", "rb") as f:
    xg_reg = pickle.load(f)
    
model = tf.keras.models.load_model("/app/case_study/stock/embedding_model.h5")
  
# Define a function to preprocess the text input
def preprocess_text():
    # Create the text input widget with the unique key
    news = st.text_input('Current News Related to Stock')

    input_seq = tokenizer.texts_to_sequences([news])

    # Pad the input sequence to match the maximum sequence length
    input_seq = pad_sequences(input_seq,maxlen=21)

    # Use the trained Word2Vec model to transform the input sequence
    input_embedding = model.predict(input_seq)
    
    return input_embedding

# Define the Streamlit app
def app():
    # Set the page title
    st.set_page_config(page_title="Stock Price Prediction using ML")

    # Set the sidebar
    st.sidebar.title("Stock Prediction App")
    st.sidebar.write("Enter the current news and price to predict the stock price.")

    # Set the main content
    st.title('Stock Price Prediction')

    # Create the input widgets
    text_input = preprocess_text()
    price_input = st.number_input('Enter the current price', value=100.0)

    # Create the prediction button
    if st.button('Predict'):
        # Preprocess the text input
        text_input_processed = xgb.DMatrix(text_input)

        # Use the pre-trained model to make a prediction
        stock_prediction = xg_reg.predict(text_input_processed)[0]

        # Predict the final stock value
        stock_price = price_input + stock_prediction

        # Display the prediction to the user
        st.write('Predicted_stock_price:', stock_price)

# Run the Streamlit app
if __name__ == '__main__':
    app()
