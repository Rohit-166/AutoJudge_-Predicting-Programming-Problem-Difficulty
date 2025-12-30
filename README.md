# AutoJudge – Predicting Programming Problem Difficulty

AutoJudge predicts the difficulty of programming problems using only textual information such as the problem statement, input description, and output description.

## Features
- Difficulty classification (Easy / Medium / Hard)
- Difficulty score prediction (1–10 scale)
- Text-only machine learning model
- Web interface using Streamlit

## Models
- **Classifier**: Logistic Regression (Accuracy ≈ 89%)
- **Regressor**: Random Forest (MAE ≈ 1.33)

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
