"""Train and evaluate all models for ML Assignment 2.

This script intentionally keeps each model definition in a separate .py file,
matching the assignment's requested model/ folder structure.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from model.logistic_regression import create_model as logistic_regression
from model.decision_tree import create_model as decision_tree
from model.knn import create_model as knn
from model.naive_bayes import create_model as naive_bayes
from model.random_forest import create_model as random_forest

ROOT = Path(__file__).resolve().parents[1]


def get_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()  # 0 = malignant, 1 = benign
    return train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    ), list(X.columns)


def main():
    (X_train, X_test, y_train, y_test), feature_names = get_data()

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    factories = {
        "Logistic Regression": logistic_regression,
        "Decision Tree": decision_tree,
        "kNN": knn,
        "Naive Bayes": naive_bayes,
        "Random Forest": random_forest,
    }

    rows = []
    for name, factory in factories.items():
        model = factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, prob),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "model_metrics.csv", index=False)
    with open(ROOT / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
