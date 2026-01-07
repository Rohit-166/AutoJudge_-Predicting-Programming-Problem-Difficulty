import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import joblib
import matplotlib.pyplot as plt
from src.preprocess import clean_text
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load vectorizer and classifier
vectorizer = joblib.load("models/vectorizer.pkl")
clf = joblib.load("models/classifier.pkl")

true = []
pred = []

# Load and predict
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
        prediction = clf.predict(X)[0]

        pred.append(prediction)
        true.append(obj["problem_class"].capitalize())

# Print metrics
print("Classifier Accuracy:", accuracy_score(true, pred))
print(classification_report(true, pred))

# Generate confusion matrix
labels = ["Easy", "Medium", "Hard"]
cm = confusion_matrix(true, pred, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Difficulty Classification")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
