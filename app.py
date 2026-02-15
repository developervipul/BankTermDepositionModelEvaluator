import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")
st.title("📊 Bank Term Deposit Subscription Predictor")

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": pickle.load(open("model/logistic_model.pkl","rb")),
        "Decision Tree": pickle.load(open("model/dt_model.pkl","rb")),
        "KNN": pickle.load(open("model/knn_model.pkl","rb")),
        "Naive Bayes": pickle.load(open("model/nb_model.pkl","rb")),
        "Random Forest": pickle.load(open("model/rf_model.pkl","rb")),
        "XGBoost": pickle.load(open("model/xgb_model.pkl","rb")),
    }
    scaler = pickle.load(open("model/scaler.pkl","rb"))
    return models, scaler

models, scaler = load_models()

# =========================
# a. Dataset Upload Option
# =========================
st.sidebar.header("Upload Test Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# =========================
# b. Model Selection Dropdown
# =========================
model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Map categorical columns to numeric
    mapping = {"yes": 1, "no": 0, "unknown": 0}  # treat 'unknown' as 0
    # Basic preprocessing (must match training preprocessing)
    df["housing"] = df["housing"].map(mapping)
    df["loan"] = df["loan"].map(mapping)
    df["y"] = df["y"].map({"yes":1, "no":0})

    X = df[["age", "campaign", "pdays", "previous", "housing", "loan",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]]
    y = df["y"]

    X_scaled = scaler.transform(X)

    try:
        X_scaled = scaler.transform(X)
    except ValueError as e:
        print(e)
        print("Columns in X:", X.columns.tolist())
        print("Columns scaler expects:", scaler.feature_names_in_)
    model = models[model_choice]
    predictions = model.predict(X_scaled)

    # =========================
    # c. Evaluation Metrics
    # =========================
    acc = accuracy_score(y, predictions)
    prec = precision_score(y, predictions)
    rec = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")

    # =========================
    # d. Confusion Matrix
    # =========================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y, predictions)
    st.text(report)

else:
    st.info("Please upload a test CSV dataset from Bank Marketing dataset.")
