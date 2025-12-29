from preprocess import load_and_preprocess
from features import extract_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df = load_and_preprocess("data/problems.csv")
X = extract_features(df, fit=False)
y = df["rating"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(Xtr, ytr)

joblib.dump(reg, "models/regressor.pkl")
print("Regressor trained & saved")
