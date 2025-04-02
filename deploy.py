import streamlit as st
import joblib  # If your models are saved as .pkl files
import numpy as np


# Load the pre-trained models and vectorizers
model_n = joblib.load('naive.pkl')  # Naive Bayes model
model_p = joblib.load('regression.pkl')  # Logistic Regression model
vectorizer_n = joblib.load('count_vectorizer.pkl')  # Vectorizer for Naive Bayes
vectorizer_p = joblib.load('tfidf_vectorizer.pkl')  # Vectorizer for Logistic Regression

def predict(text, model, vectorizer):
    # Vectorize the text input using the correct vectorizer
    text_vector = vectorizer.transform([text])  # Transform the input text
    prediction = model.predict(text_vector)  # Predict using the model
    return prediction

# Streamlit layout
st.title('Spam Email Classifier')

# Text input field
user_input = st.text_area("Enter email text here:")

# Buttons to select the model
model_choice = st.radio("Select model:", ("Model 1 (Naive Bayes)", "Model 2 (Logistic Regression)"))

# Displaying prediction result after user clicks the button
if st.button('Predict'):
    if user_input:
        if model_choice == "Model 1 (Naive Bayes)":
            result = predict(user_input, model_n, vectorizer_n)
        else:
            result = predict(user_input, model_p, vectorizer_p)
        
        if result == 1:
            st.write("Prediction: Spam")
        else:
            st.write("Prediction: Not Spam")
    else:
        st.write("Please enter some text.")
