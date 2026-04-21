import joblib
import pandas as pd

model = joblib.load("src/model/model.pkl")

def predict(data: dict):

    df = pd.DataFrame([data])

    pred = model.predict(df)[0]

    # Garantir probabilidade
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0][1]
    else:
        proba = 0.0

    return pred, proba
