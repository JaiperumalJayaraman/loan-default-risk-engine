from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "loan_default_model.joblib"


def predict_default(applicant: dict):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([applicant])
    probability = float(model.predict_proba(X)[0, 1])
    if probability < 0.35:
        risk = "Low"
    elif probability < 0.65:
        risk = "Medium"
    else:
        risk = "High"
    return risk, probability


if __name__ == "__main__":
    sample = {
        "age": 35,
        "annual_income": 75000,
        "loan_amount": 200000,
        "loan_term_months": 36,
        "credit_score": 700,
        "debt_to_income": 0.30,
        "employment_type": "Salaried",
        "payment_history": "Good",
        "home_ownership": "Own",
        "loan_purpose": "Car",
    }
    risk, probability = predict_default(sample)
    print(f"Risk: {risk}")
    print(f"Default probability: {probability:.1%}")
