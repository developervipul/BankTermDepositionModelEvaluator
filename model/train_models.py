import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import os

# Create model folder if not exists
os.makedirs("model", exist_ok=True)
# Load dataset
df = pd.read_csv("bank_data.csv")
print(df.head())
# Basic preprocessing
# Map categorical columns to numeric
mapping = {"yes": 1, "no": 0, "unknown": 0}  # treat 'unknown' as 0

df["housing"] = df["housing"].map(mapping)
df["loan"] = df["loan"].map(mapping)
df["y"] = df["y"].map({"yes": 1, "no": 0})


X = df[["age", "campaign", "pdays", "previous", "housing", "loan",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]]

y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

# 2. Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)

# 3. KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)

# 4. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# 5. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# 6. XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

# Save models
pickle.dump(log_model, open("model/logistic_model.pkl", "wb"))
pickle.dump(dt_model, open("model/dt_model.pkl", "wb"))
pickle.dump(knn_model, open("model/knn_model.pkl", "wb"))
pickle.dump(nb_model, open("model/nb_model.pkl", "wb"))
pickle.dump(rf_model, open("model/rf_model.pkl", "wb"))
pickle.dump(xgb_model, open("model/xgb_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("All 6 models trained and saved successfully.")
