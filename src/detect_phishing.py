import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def detect_phishing(text, model_path):
    model = joblib.load(model_path)
    vectorizer = TfidfVectorizer(stop_words='english')
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return prediction[0]

if __name__ == "__main__":
    text = "Your account has been compromised. Click here to reset your password."
    model_path = '../models/phishing_detector.pkl'
    result = detect_phishing(text, model_path)
    print("Phishing" if result == 1 else "Legitimate")
