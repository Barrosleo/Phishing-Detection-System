import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def detect_phishing(text, model_path):
    # Load the trained model
    model = joblib.load(model_path)
    
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Transform the input text
    features = vectorizer.transform([text])
    
    # Predict using the model
    prediction = model.predict(features)
    
    return prediction[0]

if __name__ == "__main__":
    text = "Your account has been compromised. Click here to reset your password."
    model_path = '../models/phishing_detector.pkl'
    
    # Detect phishing
    result = detect_phishing(text, model_path)
    
    # Print the result
    print("Phishing" if result == 1 else "Legitimate")

