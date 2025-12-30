import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import load_and_preprocess
from src.features import extract_features
from sklearn.ensemble import RandomForestRegressor
import joblib

df = load_and_preprocess("data/problems_data.jsonl")
X = extract_features(df, fit=False)
y = df["score"]

reg = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

reg.fit(X, y)
joblib.dump(reg, "models/regressor.pkl", compress=3)
print("Regressor trained & saved")

