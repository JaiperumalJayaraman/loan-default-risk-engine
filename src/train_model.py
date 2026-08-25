from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "loan_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_pipeline(model):
    df = pd.read_csv(DATA)
    X = df.drop(columns="default")
    y = df["default"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)]), X, y


def evaluate(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, pred, zero_division=0):.3f}")
    print(f"F1 Score : {f1_score(y_test, pred, zero_division=0):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.3f}")
    return pipeline, roc_auc_score(y_test, proba)


def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns="default")
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    }
    best, best_auc, best_name = None, -1, None
    for name, model in models.items():
        pipeline, auc = evaluate(name, *build_pipeline(model)[:1], X_train, X_test, y_train, y_test) if False else (None, None)
        pipeline, _, _ = build_pipeline(model)
        pipeline, auc = evaluate(name, pipeline, X_train, X_test, y_train, y_test)
        if auc > best_auc:
            best, best_auc, best_name = pipeline, auc, name
    joblib.dump(best, MODEL_DIR / "loan_default_model.joblib")
    print(f"\nSaved best model: {best_name} (ROC-AUC={best_auc:.3f})")


if __name__ == "__main__":
    main()
