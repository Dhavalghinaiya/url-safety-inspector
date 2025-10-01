# app.py
"""
Streamlit app for phishing URL detection using trained model.
"""

import streamlit as st
import joblib
import re

# Convert normal URL → defanged (matching dataset style)
def clean_input_url(raw_url: str) -> str:
    url = raw_url.strip()
    url = url.replace(".", "[.]")
    url = re.sub(r"^https?://", "hxxp://", url, flags=re.IGNORECASE)
    return url

# Load model + vectorizer
model = joblib.load("url_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="URL Safety Inspector", page_icon="🔒")
st.title("🔒 URL Safety Inspector")
st.write("This tool uses a model trained on the **Gamortsey/url_defanged** dataset.")

user_input = st.text_input("Enter a URL (e.g., https://example.com)")

if st.button("Check URL"):
    if not user_input:
        st.warning("⚠️ Please enter a URL")
    else:
        defanged = clean_input_url(user_input)
        X = vectorizer.transform([defanged])
        pred = model.predict(X)[0]

        if pred == 1:
            st.error("🚨 This URL is flagged as **Phishing**")
        else:
            st.success("✅ This URL appears **Safe**")

        st.write("Defanged input used for prediction:", defanged)
