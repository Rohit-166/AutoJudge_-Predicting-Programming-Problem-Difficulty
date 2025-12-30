import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import load_and_preprocess
from src.features import extract_features
from sklearn.linear_model import LogisticRegression
import joblib

df = load_and_preprocess("data/problems_data.jsonl")
X = extract_features(df, fit=True)
y = df["problem_class"]

clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
clf.fit(X, y)

joblib.dump(clf, "models/classifier.pkl", compress=3)
print("Classifier trained & saved")
