import pandas as pd
import numpy as np
import streamlit as st
import joblib


#Load out saved components
model = joblib.load('Logistic_Regression_dry_bean_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Bean Class Prediction App')
st.write('The model predicts the bean class according to the various input features below. Kindly input your values: ')


#Input features

Perimeter = st.sidebar.slider('Size/edge length: ', 538.0, 1848.0)
Eccentricity = st.sidebar.slider('Round vs long: ', 0.4, 1.0)
Solidity = st.sidebar.slider('Regular vs dented: ', 0.9, 1.0)
roundness = st.sidebar.slider('Shape type: ', 0.6, 1.0)
ShapeFactor1 = st.sidebar.slider('Compact vs elongated: ', 0.00, 0.01)
ShapeFactor2 = st.sidebar.slider('Slenderness: ', 0.00, 0.01)
ShapeFactor4 = st.sidebar.slider('Edge complexity: ', 0.98, 1.00)

#Preparing input features for the model
features = np.array([[Perimeter, Eccentricity, Solidity, roundness, ShapeFactor1, ShapeFactor2,	ShapeFactor4]])
scaled_features = scaler.transform(features)

#Prediction
if st.button('Predict Bean Class'):
    prediction_encoded = model.predict(scaled_features)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    st.success(f'Predict Bean Class: {prediction_label}')