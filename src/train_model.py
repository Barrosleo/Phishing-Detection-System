import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(features_path, labels_path):
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return model

if __name__ == "__main__":
    model = train_model('../data/preprocessed_features.csv', '../data/preprocessed_labels.csv')
    import joblib
    joblib.dump(model, '../models/phishing_detector.pkl')
