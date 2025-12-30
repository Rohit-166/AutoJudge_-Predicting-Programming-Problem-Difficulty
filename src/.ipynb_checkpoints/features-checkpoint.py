from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features(df, fit=True):
    if fit:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20000,
            stop_words="english",
            min_df=3
        )
        X = vectorizer.fit_transform(df["full_text"])
        joblib.dump(vectorizer, "models/vectorizer.pkl")
    else:
        vectorizer = joblib.load("models/vectorizer.pkl")
        X = vectorizer.transform(df["full_text"])

    return X
