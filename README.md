# Loan Default Risk Engine

## Problem Statement

Loan providers need to identify applicants who may have a higher probability of defaulting on a loan. This project builds a simple machine learning risk engine that uses applicant and loan information to predict whether a borrower is likely to default.

The goal is to demonstrate a complete beginner-friendly machine learning workflow from raw data to a usable prediction interface.

## Approach

1. Load the loan application dataset.
2. Perform basic data cleaning and exploratory analysis.
3. Separate input features and the default target.
4. Encode categorical variables and scale numerical variables where required.
5. Split the data into training and testing sets.
6. Train a Logistic Regression model as a simple baseline.
7. Train a Random Forest model for a non-linear comparison.
8. Evaluate the models using accuracy, precision, recall, F1-score and ROC-AUC.
9. Save the better model as a reusable pipeline.
10. Use Streamlit to provide a simple loan-risk prediction interface.

## Key Insights

- Higher debt-to-income ratios generally indicate greater repayment pressure.
- Lower credit scores are associated with higher default risk.
- Longer loan terms can increase repayment exposure.
- Previous payment history is an important indicator of default risk.
- Model probabilities are used to place applications into Low, Medium or High risk categories.

> This is an educational project and should not be used as a real lending or credit-decision system.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/JaiperumalJayaraman/loan-default-risk-engine.git
cd loan-default-risk-engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train_model.py
```

This creates the trained model inside `models/` and prints evaluation metrics.

### 4. Run the prediction app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Project Structure

```text
loan-default-risk-engine/
├── data/
│   └── loan_data.csv
├── models/
│   └── loan_default_model.joblib
├── src/
│   ├── train_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```
