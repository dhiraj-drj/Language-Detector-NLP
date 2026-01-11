import streamlit as st
import pandas as pd
import joblib

# Custom CSS for better coloring and readability
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* A light, neutral background */
    }
    .main-title {
        color: #8B4513; /* Dark Brown color */
    }
    h1, .stTextArea > label {
        color: #31333f; /* Dark text for good contrast */
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    # Load the pre-trained model and vectorizer
    loaded_model = joblib.load('NB_Language_Detector.pkl')
    loaded_vectorizer = joblib.load('vectorizer.pkl')
    expected_features = joblib.load('features.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading a required file: {e}. Please make sure all .pkl files are in the correct directory.")
    st.stop()

# Streamlit App Title
st.markdown('<h1 class="main-title">Language Detector</h1>', unsafe_allow_html=True)

# Input text area
user_input = st.text_area('Enter text here to detect its language:', "", height=150)

if st.button('Detect Language'):
    if user_input:
        # Transform the input text using the loaded vectorizer
        vectorized_input = loaded_vectorizer.transform([user_input])
        
        # Create a DataFrame with the vectorized input and the feature names from the vectorizer
        input_df = pd.DataFrame(vectorized_input.toarray(), columns=loaded_vectorizer.get_feature_names_out())
        
        # Align the input DataFrame with the expected features
        data = pd.DataFrame(columns=expected_features).add(input_df, fill_value=0)[expected_features]
        
        # Predict the language and display the result
        output = loaded_model.predict(data)
        st.success(f'Predicted Language: **{output[0]}**')
    else:
        st.warning('Please enter some text to detect its language.')
