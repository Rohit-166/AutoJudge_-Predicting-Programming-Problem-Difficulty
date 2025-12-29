from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import joblib
import numpy as np
import pandas as pd

def extract_features(df, fit=True):
    if fit:
        vectorizer = CountVectorizer()
        X_tags = vectorizer.fit_transform(df["tags"])
        joblib.dump(vectorizer, "models/vectorizer.pkl")
    else:
        vectorizer = joblib.load("models/vectorizer.pkl")
        X_tags = vectorizer.transform(df["tags"])

    solved = pd.to_numeric(
        df["Solved"],
        errors="coerce"   
    ).fillna(0).values.reshape(-1, 1)

    return hstack([X_tags, solved])
