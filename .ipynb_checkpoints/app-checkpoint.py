import streamlit as st
import joblib
from src.preprocess import clean_text

vectorizer = joblib.load("models/vectorizer.pkl")
reg = joblib.load("models/regressor.pkl")

# Thresholds from TRUE score distribution
EASY_TH = 3.9
HARD_TH = 6.3

def score_to_class(score):
    if score < EASY_TH:
        return "Easy"
    elif score < HARD_TH:
        return "Medium"
    else:
        return "Hard"

st.title("AutoJudge – Predicting Programming Problem Difficulty")
st.caption("Predicts difficulty class and score from problem text")

desc = st.text_area("Problem Description")
inp = st.text_area("Input Description")
out = st.text_area("Output Description")

if st.button("Predict"):
    text = clean_text(desc + " " + inp + " " + out)
    X = vectorizer.transform([text])

    score = reg.predict(X)[0]
    score = max(1.0, min(10.0, score))

    st.success(f"Predicted Difficulty Class: {score_to_class(score)}")
    st.info(f"Predicted Difficulty Score (1–10): {score:.2f}")
