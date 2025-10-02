# safe_browsing.py

import requests
import os

# IMPORTANT: Your API key should be stored as an environment variable, not written here.
# Before running the app, set the variable in your terminal.
API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

def check_url_google(url):
    """
    Check a URL with Google Safe Browsing API.
    Returns: dict { 'malicious': bool, 'threat': str }
    """
    if not API_KEY:
        print("Warning: GOOGLE_SAFE_BROWSING_API_KEY environment variable not set. Skipping check.")
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
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        if "matches" in data:
            return {"malicious": True, "threat": data["matches"][0]["threatType"]}
    except requests.exceptions.RequestException as e:
        print(f"Error calling Google Safe Browsing API: {e}")
        return {"malicious": False, "threat": "API_ERROR"}

    return {"malicious": False, "threat": None}