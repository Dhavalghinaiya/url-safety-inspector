# app.py
from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safe_browsing import check_url_google
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# --- SECTION 1: MODEL LOADING AND ON-DEMAND TRAINING ---

# --- Model 1: URL Classifier (Transformers) ---
print("Loading URL classification model...")
URL_MODEL_NAME = "hadimhd/bert-phishing-links-classifier"
url_tokenizer = AutoTokenizer.from_pretrained(URL_MODEL_NAME)
url_model = AutoModelForSequenceClassification.from_pretrained(URL_MODEL_NAME)
print("URL Model loaded.")

# --- Model 2: Email Classifier (Scikit-learn) ---
EMAIL_MODEL_DIR = "./email_classifier_model_sklearn"
MODEL_PATH = os.path.join(EMAIL_MODEL_DIR, 'model.joblib')
VECTORIZER_PATH = os.path.join(EMAIL_MODEL_DIR, 'vectorizer.joblib')

# This function contains all the training logic.
def train_and_save_model():
    """
    Trains a new email classification model using Scikit-learn and saves it.
    This function is called automatically if a trained model is not found.
    """
    print("--- Starting one-time training for the email model... ---")
    
    # 1. DATA PREPARATION
    try:
        df = pd.read_csv("se_phishing_test_set.csv")
    except FileNotFoundError:
        print("FATAL ERROR: 'se_phishing_test_set.csv' not found. Cannot train email model.")
        return False

    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)
    df.dropna(subset=['email_text'], inplace=True)
    df['label'] = df['label'].map({'Benign': 0, 'Malicious': 1})
    
    X_train, _, y_train, _ = train_test_split(df['email_text'], df['label'], test_size=0.2, random_state=42)
    print("Data prepared.")

    # 2. FEATURE EXTRACTION (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print("Text features created.")

    # 3. MODEL TRAINING
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    # 4. SAVE THE MODEL AND VECTORIZER
    os.makedirs(EMAIL_MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"Model and vectorizer saved to '{EMAIL_MODEL_DIR}'.")
    return True

# Check if the email model exists. If not, train it.
if not os.path.exists(MODEL_PATH):
    print("Email classifier model not found.")
    train_and_save_model()

# Now, load the email model (it's guaranteed to exist at this point).
email_model_loaded = False
try:
    print("Loading Scikit-learn Email classification model...")
    email_vectorizer = joblib.load(VECTORIZER_PATH)
    email_model = joblib.load(MODEL_PATH)
    print("Scikit-learn Email Model loaded successfully.")
    email_model_loaded = True
except (OSError, FileNotFoundError):
    print("!!! WARNING: Failed to load the Scikit-learn email model.")


# --- SECTION 2: FLASK ROUTES AND APPLICATION LOGIC ---

GUIDANCE = {
    "email": [
        "Do not click links or download attachments.",
        "Verify the sender’s email address carefully.",
        "Report the email to your university IT/security team.",
        "Visit the official website by typing it yourself."
    ]
}

@app.route("/", methods=["GET", "POST"])
def index():
    result, steps, result_class = None, None, None
    analysis_type = "url"

    if request.method == "POST":
        # A) URL Inspector Logic
        if "url" in request.form:
            analysis_type = "url"
            url = request.form["url"]
            inputs = url_tokenizer(url, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = url_model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            phishing_prob = probabilities[0][1].item()
            confidence_phishing = round(phishing_prob * 100, 2)
            is_phishing_pred = phishing_prob > 0.5
            google_check = check_url_google(url)
            is_malicious = is_phishing_pred or google_check["malicious"]

            if is_malicious:
                if confidence_phishing >= 70 or google_check["malicious"]:
                    result_class, result = 'danger', f"🚨 High-Risk Link Detected (Confidence: {confidence_phishing}%)"
                else:
                    result_class, result = 'warning', f"🤔 Suspicious Link Detected (Confidence: {confidence_phishing}%)"
                if google_check["malicious"]:
                    result += f" [Google: {google_check['threat']}]"
                steps = GUIDANCE["email"]
            else:
                result_class, result = 'safe', f"✅ Link Appears Safe (Confidence: {100 - confidence_phishing:.2f}%)"

        # B) Email Inspector Logic
        elif "email_text" in request.form:
            analysis_type = "email"
            if not email_model_loaded:
                result_class, result = 'warning', "⚠️ Email analysis is disabled."
                steps = ["The email classification model could not be loaded."]
            else:
                email_content = request.form["email_text"]
                email_tfidf = email_vectorizer.transform([email_content])
                prediction = email_model.predict(email_tfidf)[0]
                proba = email_model.predict_proba(email_tfidf)[0]
                confidence = round(max(proba) * 100, 2)

                if prediction == 1: # 1 = Malicious
                    result_class, result = 'danger', f"🚨 This email shows signs of phishing. (Confidence: {confidence}%)"
                    steps = GUIDANCE["email"]
                else: # 0 = Benign
                    result_class, result = 'safe', f"✅ This email appears to be safe. (Confidence: {confidence}%)"

    return render_template("index.html", result=result, result_class=result_class, steps=steps, analysis_type=analysis_type)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

