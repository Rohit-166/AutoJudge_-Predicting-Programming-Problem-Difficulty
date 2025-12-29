import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

 
    df["tags"] = (
        df["Problem_tag_1"].fillna("") + " " +
        df["Problem_tag_2"].fillna("") + " " +
        df["Problem_tag_3"].fillna("") + " " +
        df["Problem_tag_4"].fillna("")
    )

   
    df["rating"] = pd.to_numeric(
        df["Problem Rating"],
        errors="coerce"
    )

 
    df = df.dropna(subset=["rating"])

    def label_class(r):
        if r < 1200:
            return "Easy"
        elif r < 2000:
            return "Medium"
        else:
            return "Hard"

    df["problem_class"] = df["rating"].apply(label_class)

    return df

