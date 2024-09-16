import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].str.lower()
    vectorizer = TfidfVectorizer(stop_words='english')
    features = vectorizer.fit_transform(data['text'])
    return features, data['label']

if __name__ == "__main__":
    features, labels = preprocess_data('../data/emails.csv')
    pd.DataFrame(features.toarray()).to_csv('../data/preprocessed_features.csv', index=False)
    pd.DataFrame(labels).to_csv('../data/preprocessed_labels.csv', index=False)
