import json
import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_preprocess(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    df["full_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["input_description"].fillna("") + " " +
        df["output_description"].fillna("")
    ).apply(clean_text)

    df["score"] = pd.to_numeric(df["problem_score"], errors="coerce")
    df["problem_class"] = df["problem_class"].str.capitalize()

    df = df.dropna(subset=["score", "problem_class"])
    return df
