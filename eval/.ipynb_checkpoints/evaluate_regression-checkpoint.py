import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import joblib
import numpy as np
from src.preprocess import clean_text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

vectorizer = joblib.load("models/vectorizer.pkl")
reg = joblib.load("models/regressor.pkl")

y_true, y_pred = [], []

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
        y_pred.append(reg.predict(X)[0])
        y_true.append(obj["problem_score"])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("MAE :", mean_absolute_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R²  :", r2_score(y_true, y_pred))
