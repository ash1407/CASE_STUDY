import streamlit as st
import numpy as np
import pickle
import xgboost as xgb
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# loading the saved models


xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, seed=123)
#xg_reg.load_model("C:\\Users\\qj771f\\Desktop\\Boeing\\Resources\\4_ipynb files\\7_web app\\Streamlit\\stocks\\model_tfidf.json")
#xgb_model  = json.loads(open('/app/case_study/stock/model_tfidf.json', 'r').read())
xgb_model.load_model('/app/case_study/stock/model_tfidf.json')

# Load the pre-trained tf-idf vectorizer
with open('/app/case_study/stock/tfidf.pickle', 'rb') as f:
    tfidf = pickle.load(f)
    
# Define a function to preprocess the text input

def preprocess_text():
    # Create the text input widget with the unique key
    news = st.text_input('Current News Related to Stock')
    
    # Preprocess the text input using the pre-trained tf-idf vectorizer
    preprocessed_input = tfidf.transform([news])
    
    return preprocessed_input

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
        #preprocessed_input = tfidf.transform([preprocess_text()])
        
        # Use the pre-trained model to make a prediction
        stock_prediction = xgb_model.predict(preprocessed_input)

        # Predict final stock value
        stock_price = current_price + stock_prediction[0]
        
        # Display the prediction to the user
        st.write('Prediction:', stock_price)

# Run the Streamlit app
if __name__ == '__main__':
    app()
	
