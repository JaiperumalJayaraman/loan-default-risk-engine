from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "loan_default_model.joblib"

st.set_page_config(page_title="Loan Default Risk Engine", page_icon="💳", layout="centered")
st.title("💳 Loan Default Risk Engine")
st.caption("Educational ML demo — not a real lending decision system.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Applicant Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 80, 35)
    annual_income = st.number_input("Annual Income", 10000, 500000, 75000, step=5000)
    loan_amount = st.number_input("Loan Amount", 10000, 2000000, 200000, step=10000)
    loan_term_months = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)
    credit_score = st.slider("Credit Score", 300, 850, 700)
with col2:
    debt_to_income = st.slider("Debt-to-Income Ratio", 0.05, 0.85, 0.30, 0.01)
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Business"])
    payment_history = st.selectbox("Payment History", ["Good", "Average", "Poor"])
    home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
    loan_purpose = st.selectbox("Loan Purpose", ["Education", "Home", "Car", "Personal", "Medical"])

if st.button("Predict Default Risk", type="primary"):
    applicant = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "employment_type": employment_type,
        "payment_history": payment_history,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
    }])
    probability = float(model.predict_proba(applicant)[0, 1])
    if probability < 0.35:
        risk = "LOW RISK"
        st.success(f"{risk} — estimated default probability: {probability:.1%}")
    elif probability < 0.65:
        risk = "MEDIUM RISK"
        st.warning(f"{risk} — estimated default probability: {probability:.1%}")
    else:
        risk = "HIGH RISK"
        st.error(f"{risk} — estimated default probability: {probability:.1%}")
    st.progress(probability, text=f"Default probability: {probability:.1%}")
