import pickle
import json
import streamlit as st
import numpy as np

# loading the saved models
Stock_price_model = json.loads(open('/app/case_study/stock/model.json', 'r').read())

# Load the pre-trained tf-idf vectorizer
with open('/app/case_study/stock/tfidf.pickle', 'rb') as f:
    tfidf = pickle.load(f)
    
# Define a function to preprocess the text input
def preprocess_text():
	

    # Create the text input widget with the unique key
    #news = st.text_input("Current News Related to Stock", "")	
    news = st.text_input('Current News Related to Stock',"")
    return news

# Define the Streamlit apps
def app():
    # page title
    st.title('Stock Price Prediction using ML')

    # Text input
    text_input = preprocess_text()

    # Numeric input
    current_price = st.number_input('Current Price')

    # When the user clicks the 'Predict' button, preprocess the input and pass it to the model
    if st.button('Predict'):
        # Preprocess the text input
        preprocessed_input = tfidf.transform([preprocess_text()])
        
        # Use the pre-trained model to make a prediction
        stock_prediction = xgb_model.predict(preprocessed_input)

        # Predict final stock value
        stock_price = current_price + stock_prediction[0]
        

        # Display the prediction to the user
        st.write('Prediction:', stock_price)

# Run the Streamlit app
if __name__ == '__main__':
    app()
	
