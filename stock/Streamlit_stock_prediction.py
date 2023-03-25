# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:01:15 2022

@author: siddhardhan
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

Stock_price_model = pickle.load(open('model_tfidf.json', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Stock Price Prediction',
                          
                          ['Stock Prediction'],
                          icons=['activity'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Stock Price Prediction'):
    
    # page title
    st.title('Stock Price Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2 = st.columns(2)
    
    with col1:
        current_price = st.text_input('Current Price')
        
    with col2:
        news = st.text_input('Current News Related to Stock')
    
    
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        stock_prediction = Stock_price_model.predict([[news]])
        stock_price = current_price + stock_prediction
        
    st.success('Stock_Price :',stock_price)




















