# AutoJudge – Predicting Programming Problem Difficulty

## 📌 Project Overview

AutoJudge is a machine learning–based system that predicts the difficulty of programming problems using only their textual descriptions.

The project performs:
- **Classification** of problems into **Easy / Medium / Hard**
- **Regression** to predict a **numerical difficulty score (1–10)**

The system also includes a **web-based interface** that allows users to input a programming problem and instantly receive predictions.

---

## 📊 Dataset Used

The dataset contains programming problems with the following fields:
- Problem statement  
- Input description  
- Output description  
- Difficulty label (Easy / Medium / Hard)  
- Numerical difficulty score  

All textual fields are combined and processed to extract meaningful features for model training.

---

## 🧠 Approach & Models Used

### 1️⃣ Data Preprocessing
- Text cleaning (lowercasing, punctuation removal)
- Stopword removal
- Combining problem statement, input, and output into a single text feature

### 2️⃣ Feature Extraction
- **TF-IDF Vectorization** is used to convert textual data into numerical feature vectors.

### 3️⃣ Models Used

| Task | Model |
|----|----|
| Difficulty Classification | Logistic Regression |
| Difficulty Score Prediction | Random Forest Regressor |

---

## 📈 Evaluation Metrics

| Task | Metric | Result |
|----|----|----|
| Classification | Accuracy | ~89% |
| Regression | Mean Absolute Error (MAE) | ~1.33 |

---

## 🌐 Web Interface Explanation

The project includes a **Streamlit-based web application** that allows users to:
1. Enter a programming problem description  
2. Click **Predict**  
3. View:
   - Difficulty category (Easy / Medium / Hard)
   - Predicted difficulty score (1–10)

The interface runs locally and loads pre-trained models. No external hosting is required.

---

## ▶️ Steps to Run the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Rohit-166/AutoJudge_-Predicting-Programming-Problem-Difficulty.git
cd AutoJudge_-Predicting-Programming-Problem-Difficulty
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Web Application
```bash
streamlit run app.py
```



---

## 💾 Saved Trained Models

The repository includes pre-trained models:
- **Classification model** (Easy / Medium / Hard)
- **Regression model** (difficulty score)

These models are stored in the `models/` directory and are directly loaded by the web application.

---

## 🎥 Demo Video 

📌 **Demo Video Link:**  
👉 *Add your 2–3 minute demo video link here (Google Drive / YouTube Unlisted)*

---

## 🌐 Live Streamlit App

🔗 **[Click here to open the Streamlit App](https://predicting-programming-problem-difficulty.streamlit.app/)**




