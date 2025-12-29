from preprocess import load_and_preprocess
from features import extract_features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

df = load_and_preprocess("data/problems.csv")
X = extract_features(df, fit=True)
y = df["problem_class"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, ytr)

joblib.dump(clf, "models/classifier.pkl")
print("Classifier trained & saved")
