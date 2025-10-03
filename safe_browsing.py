# safe_browsing.py
import requests
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Your API key is loaded from the .env file for security
API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

def check_url_google(url):
    """Checks a URL with the Google Safe Browsing API."""
    if not API_KEY:
        print("Warning: GOOGLE_SAFE_BROWSING_API_KEY not set. Skipping check.")
        return {"malicious": False, "threat": None}

    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={API_KEY}"
    body = {
        "client": {"clientId": "url-safety-inspector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "PHISHING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    
    try:
        response = requests.post(endpoint, json=body)
        response.raise_for_status()
        data = response.json()
        if "matches" in data:
            return {"malicious": True, "threat": data["matches"][0]["threatType"]}
    except requests.exceptions.RequestException as e:
        print(f"Error calling Google Safe Browsing API: {e}")
        return {"malicious": False, "threat": "API_ERROR"}

    return {"malicious": False, "threat": None}