from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from model.logistic_regression import create_model as logistic_regression
from model.decision_tree import create_model as decision_tree
from model.knn import create_model as knn
from model.naive_bayes import create_model as naive_bayes
from model.random_forest import create_model as random_forest

ROOT = Path(__file__).resolve().parent

MODEL_FACTORIES = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "kNN": knn,
    "Naive Bayes": naive_bayes,
    "Random Forest": random_forest,
}

st.set_page_config(
    page_title="Breast Cancer Classification - ML Assignment 2",
    layout="wide",
)


@st.cache_resource(show_spinner="Training selected model...")
def train_model(model_name):
    """Train the requested model on the assignment training split."""
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    model = MODEL_FACTORIES[model_name]()
    model.fit(X_train, y_train)
    return model, list(X.columns)


st.title("Machine Learning Assignment 2 - Classification Model Explorer")
st.caption(
    "Dataset: Breast Cancer Wisconsin (Diagnostic) | "
    "Target: 0 = Malignant, 1 = Benign"
)

st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FACTORIES.keys()))
uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

st.info(
    "Upload the supplied test_data.csv. It contains the 30 feature columns "
    "and the target column required to calculate evaluation metrics."
)

if uploaded is None:
    st.warning("Upload test_data.csv from the repository to display model results.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()

if "target" not in df.columns:
    st.error("The uploaded CSV must contain a 'target' column for evaluation.")
    st.stop()

model, expected_features = train_model(model_name)
missing_features = [col for col in expected_features if col not in df.columns]
extra_features = [
    col for col in df.columns
    if col not in expected_features and col != "target"
]

if missing_features:
    st.error("Missing required feature columns: " + ", ".join(missing_features))
    st.stop()

if extra_features:
    st.warning(
        "Extra columns were ignored: " + ", ".join(extra_features)
    )

X = df[expected_features]
y = df["target"]

try:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, prob),
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1 Score": f1_score(y, pred, zero_division=0),
    "MCC Score": matthews_corrcoef(y, pred),
}

st.subheader(f"Evaluation Metrics - {model_name}")
metric_columns = st.columns(3)
for i, (label, value) in enumerate(metrics.items()):
    metric_columns[i % 3].metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Malignant (0)", "Actual Benign (1)"],
        columns=["Pred Malignant (0)", "Pred Benign (1)"],
    )
    st.dataframe(cm_df, use_container_width=True)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y,
        pred,
        labels=[0, 1],
        target_names=["Malignant", "Benign"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("Sample Predictions")
preview = df.copy()
preview["predicted_target"] = pred
preview["probability_benign"] = prob
st.dataframe(preview.head(20), use_container_width=True)
