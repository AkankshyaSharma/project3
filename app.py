import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# UI
st.title("Sentiment Analysis on Product Reviews")

st.write("Enter a product review to check sentiment")

# Input box
review = st.text_area("Enter your review here")

# Button
if st.button("Predict Sentiment"):
    if review.strip() != "":
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        st.write("Prediction:")

        if prediction[0] == "Positive":
            st.success("Positive")
        elif prediction[0] == "Negative":
            st.error("Negative")
        else:
            st.warning("Neutral")
    else:
        st.write("Please enter some text")