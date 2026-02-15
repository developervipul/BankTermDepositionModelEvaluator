import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef
)
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
print(sklearn.__version__)
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
# Sidebar Options
# =========================
st.sidebar.header("Upload Test Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

# =========================
# Download Sample Dataset
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Download Sample Dataset")

try:
    with open("bank_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="⬇ Download Sample CSV",
            data=file,
            file_name="bank_data.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.sidebar.warning("Sample dataset file not found in main project folder.")

# =========================
# Main App
# =========================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Check required columns
    required_cols = ["housing", "loan", "y"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    # Map categorical columns
    mapping = {"yes": 1, "no": 0, "unknown": 0}
    df["housing"] = df["housing"].map(mapping)
    df["loan"] = df["loan"].map(mapping)
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    # Feature selection (must match training)
    feature_columns = [
        "age", "campaign", "pdays", "previous", "housing", "loan",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed"
    ]

    for col in feature_columns:
        if col not in df.columns:
            st.error(f"Missing feature column: {col}")
            st.stop()

    X = df[feature_columns]
    y = df["y"]

    # Scale features
    try:
        X_scaled = scaler.transform(X)
    except ValueError as e:
        st.error("Feature mismatch between uploaded dataset and trained scaler.")
        st.write("Error:", e)
        st.write("Columns in uploaded data:", X.columns.tolist())
        st.write("Columns expected by scaler:", scaler.feature_names_in_)
        st.stop()

    # Model prediction
    model = models[model_choice]
    predictions = model.predict(X_scaled)

    # =========================
    # Evaluation Metrics
    # =========================
    acc = accuracy_score(y, predictions)
    prec = precision_score(y, predictions, zero_division=0)
    rec = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    mcc = matthews_corrcoef(y, predictions)

    # AUC (if available)
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_scaled)
        else:
            y_proba = None

        if y_proba is not None:
            auc = roc_auc_score(y, y_proba)
        else:
            auc = None

    except Exception:
        auc = None

    # =========================
    # Display Metrics
    # =========================
    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("MCC Score", f"{mcc:.3f}")

    if auc is not None:
        col6.metric("AUC Score", f"{auc:.3f}")
    else:
        col6.metric("AUC Score", "N/A")

    # =========================
    # Confusion Matrix
    # =========================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # =========================
    # Classification Report
    # =========================
    st.subheader("Classification Report")
    report = classification_report(y, predictions)
    st.text(report)

else:
    st.info("Please upload a test CSV dataset from Bank Marketing dataset.")
