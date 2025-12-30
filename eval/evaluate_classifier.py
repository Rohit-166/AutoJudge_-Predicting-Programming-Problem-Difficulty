import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import joblib
from src.preprocess import clean_text
from sklearn.metrics import classification_report, accuracy_score

vectorizer = joblib.load("models/vectorizer.pkl")
clf = joblib.load("models/classifier.pkl")

true, pred = [], []

with open("data/problems_data.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        text = clean_text(
            obj["title"] + " " +
            obj["description"] + " " +
            obj["input_description"] + " " +
            obj["output_description"]
        )
        X = vectorizer.transform([text])
        pred.append(clf.predict(X)[0])
        true.append(obj["problem_class"].capitalize())

print("Classifier Accuracy:", accuracy_score(true, pred))
print(classification_report(true, pred))
