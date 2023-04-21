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
import streamlit as st
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer()

# loading the saved models
xg_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, seed=123)

with open("/app/case_study/stock/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f) 

# Load the model from a local file
with open("/app/case_study/stock/xg_reg_model.pickle", "rb") as f:
    xg_reg = pickle.load(f)


with open("/app/case_study/stock/X_tfidf.pickle", "rb") as f:
    X_tfidf=pickle.load(f)
 
with open("/app/case_study/stock/vectorizer.pickle", "rb") as f:
    vectorizer=pickle.load(f)


# Load the pre-trained model
model = tf.keras.models.load_model("/app/case_study/stock/embedding_model.h5")

# Define a function to preprocess the text input
def preprocess_text():
    # Create the text input widget with the unique key
    news = st.text_input('Current News Related to Stock')
    
    # Tokenize and pad the text input
    sequences = tokenizer.texts_to_sequences([news])
    X1 = pad_sequences(sequences, maxlen=24)
    
    # Transform the text to numeric vectors using Word2Vec
    embeddings1 = model.predict(X1)
    
    # Transform the text to numeric vectors using tf-idf
    X_tfidf = vectorizer.transform([news]).toarray()
    
    # Concatenate the two sets of embeddings
    embeddings = np.concatenate([embeddings1, X_tfidf], axis=1)
    
    return embeddings

import streamlit as st
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto']

# Define Streamlit app
def app():
    
    # page title
    st.title('Stock Price Prediction using ML')
    
    # Set the sidebar
    st.sidebar.title("Stock Prediction App")
    st.sidebar.write("Enter the current news and price to predict the stock price.")
    
    text = preprocess_text()

    # Numeric input title
    current_price = st.number_input('Enter current stock price:')

    # When the user clicks the 'Predict' button, preprocess the input and pass it to the model
    if st.button('Predict'):

        # Use the pre-trained model to make a prediction
        stock_prediction = xg_reg.predict(text)

        # Display the prediction to the user
        st.subheader('Percentage change in current stock price will be:')
        st.write("{:.2f}%".format(float(stock_prediction)))

        # Display whether the predicted stock price is negative or positive
        st.subheader('Predicted change in stock price:')
        if stock_prediction[0] > 0:
            st.write('The stock price will increase.')
        elif stock_prediction[0] < 0:
            st.write('The stock price will decrease.')
        else:
            st.write('The predicted stock price is unchanged.')

        # Predict final stock value
        stock_price = current_price + (stock_prediction[0]/100)*current_price 

        # Display the predicted stock price to the user
        st.subheader('Predicted stock price:')
        st.write("Rs: {:,.2f}".format(stock_price))

# Run the Streamlit app
if __name__ == '__main__':
    app()
