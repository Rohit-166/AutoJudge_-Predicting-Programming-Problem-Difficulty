import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack

vectorizer = joblib.load("models/vectorizer.pkl")
clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")

st.title("AutoJudge – Difficulty Predictor (Tag Based)")

tag1 = st.text_input("Problem Tag 1")
tag2 = st.text_input("Problem Tag 2")
tag3 = st.text_input("Problem Tag 3")
tag4 = st.text_input("Problem Tag 4")
solved = st.number_input("Solved Count", min_value=0)

if st.button("Predict"):
    tags = " ".join([tag1, tag2, tag3, tag4])
    X_tags = vectorizer.transform([tags])
    X = hstack([X_tags, [[solved]]])

    st.success(f"Difficulty Class: {clf.predict(X)[0]}")
    st.info(f"Difficulty Score: {reg.predict(X)[0]:.1f}")
