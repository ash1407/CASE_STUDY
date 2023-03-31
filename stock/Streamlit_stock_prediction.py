import pickle
import json
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

#Stock_price_model = pickle.load(open('/app/case_study/stock/model.json', 'r'))

Stock_price_model = json.loads(open('/app/case_study/stock/model.json', 'r').read())

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Stock Price Prediction',
                          
                          ['Stock Prediction'],
                          icons=['activity'],
                          default_index=0)
    
    


# page title
st.title('Stock Price Prediction using ML')

# Define a function to preprocess the text input
def preprocess_text(text):
    current_price = st.text_input('Current Price')
    news = st.number_input('Current News Related to Stock')
    return current_price,news

# Define the Streamlit app
def app():
    
    # Add a text input field for the user to enter their input
    text_input = st.text_input('Enter your text input:')
    
    # When the user clicks the 'Predict' button, preprocess the input and pass it to the model
    if st.button('Predict'):
        # Preprocess the text input
        preprocessed_input = preprocess_text(text_input)
        
        # Convert the preprocessed input to a numpy array
        input_array = np.array([preprocessed_input])
        
        # Use the pre-trained model to make a prediction
        stock_prediction = Stock_price_model.predict(input_array)
	stock_price = current_price + stock_prediction
        
        # Display the prediction to the user
        st.write('Prediction:', stock_price)
        
# Run the app
if __name__ == '__main__':
    app()
