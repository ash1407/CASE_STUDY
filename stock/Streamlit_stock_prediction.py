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
    news = st.text_input('Current News Related to Stock')
    return current_price

# Define the Streamlit app
def app():
    
    
    # When the user clicks the 'Predict' button, preprocess the input and pass it to the model
    if st.button('Predict'):
        # Preprocess the text input
        preprocessed_input = preprocess_text(text_input)
	
	#Numeric input
	current_price = st.number_input('Current Price')
        
        # Convert the preprocessed input to a numpy array
        input_array = np.array([preprocessed_input])
        
        # Use the pre-trained model to make a prediction
        stock_prediction = Stock_price_model.predict(input_array)
	
	#prdict final stock value
        stock_price = current_price + stock_prediction
        
        # Display the prediction to the user
        st.write('Prediction:', stock_price)
        
# Run the app
if __name__ == '__main__':
    app()
