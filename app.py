# app.py

from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safe_browsing import check_url_google

app = Flask(__name__)

# --- Step 1: Load the pre-trained model and tokenizer ---
print("Loading pre-trained BERT model...")
MODEL_NAME = "hadimhd/bert-phishing-links-classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")

# --- Guidance steps (no changes here) ---
GUIDANCE = {
    "email": [
        "Do not click links or download attachments.",
        "Verify the sender’s email address carefully.",
        "Report the email to your university IT/security team.",
        "Visit the official website by typing it yourself."
    ],
    "sms": [
        "Do not tap the link.",
        "Block the sender.",
        "Report spam to your carrier."
    ],
    "clicked": [
        "Disconnect from the internet immediately.",
        "Run an antivirus scan.",
        "Change your passwords from a secure device."
    ]
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    steps = None

    if request.method == "POST":
        url = request.form["url"]
        channel = request.form.get("channel", "email")

        # --- Layer 1: BERT model prediction ---
        inputs = tokenizer(url, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probabilities, dim=-1).item()
        confidence = round(probabilities[0][pred].item() * 100, 2)
        
        # For this model, 1 = phishing
        ml_flag = (pred == 1)

        # --- Layer 2: Google Safe Browsing check ---
        google_check = check_url_google(url)

        # --- Final Decision: Combine results ---
        if ml_flag or google_check["malicious"]:
            result = f"🚨 Suspicious link detected (BERT model confidence: {confidence}%)"
            if google_check["malicious"]:
                result += f" [Google Safe Browsing: {google_check['threat']}]"
            steps = GUIDANCE[channel]
        else:
            result = f"✅ Link appears safe (BERT model confidence: {confidence}%)"
            steps = None

    return render_template("index.html", result=result, confidence=confidence, steps=steps)

if __name__ == "__main__":
    app.run(debug=True)