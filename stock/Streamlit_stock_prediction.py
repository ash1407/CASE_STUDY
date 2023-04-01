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

# Define the Streamlit apps
def app():
    # page title
    st.title('Stock Price Prediction using ML')
    
    # Set the sidebar
    st.sidebar.title("Stock Prediction App")
    st.sidebar.write("Enter the current news and price to predict the stock price.")

    # Text input
    text= preprocess_text()

    # Numeric input
    current_price = st.number_input('Current Price')

    # When the user clicks the 'Predict' button, preprocess the input and pass it to the model
    if st.button('Predict'):
        # Preprocess the text input
        
        # Use the pre-trained model to make a prediction
        stock_prediction = xg_reg.predict(text.reshape(-1, 1))[0]

        # Predict final stock value
        stock_price = current_price + stock_prediction
        
        # Display the prediction to the user
        st.write('Prediction:', stock_price)

# Run the Streamlit app
if __name__ == '__main__':
    app()
