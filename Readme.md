# Bank Marketing ML Streamlit App
bank_data.csv present in repo. for data upload
## Description
This project predicts whether a bank customer will subscribe to a term deposit. 
It implements **6 machine learning classification models**:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- XGBoost

The app allows:
- Uploading a test CSV dataset  
- Selecting any of the 6 models
- Displaying evaluation metrics (Accuracy, Precision, Recall, F1)
- Viewing confusion matrix and classification report

## Project Structure
Bank-Marketing-ML-Streamlit-App/
│── .DS_Store
├── app.py                     # Main Streamlit application
├── bank_data.csv              # Dataset file
├── requirements.txt           # Project dependencies
├── models/                    # Saved trained ML models (if applicable)
├── models/                    # Stored trained models & preprocessing objects
│   ├── dt_model.pkl           # Decision Tree model
│   ├── knn_model.pkl          # K-Nearest Neighbors model
│   ├── logistic_model.pkl     # Logistic Regression model
│   ├── nb_model.pkl           # Naive Bayes model
│   ├── rf_model.pkl           # Random Forest model
│   ├── xgb_model.pkl          # XGBoost model
│   └── scaler.pkl             # Feature scaler

## c) Machine Learning Model Performance Comparison

| ML Model Name              | Accuracy | Precision | Recall | F1 Score | MCC   | AUC  |
|----------------------------|----------|-----------|--------|----------|-------|------|
| Logistic Regression        | 0.899    | 0.704     | 0.185  | 0.293    | 0.327 | 0.761 |
| Decision Tree              | 0.947    | 0.847     | 0.647  | 0.733    | 0.713 | 0.914 |
| k-Nearest Neighbors (kNN)  | 0.908    | 0.680     | 0.349  | 0.462    | 0.445 | 0.876 |
| Naive Bayes                | 0.831    | 0.334     | 0.500  | 0.401    | 0.315 | 0.757 |
| Random Forest (Ensemble)   | 0.951    | 0.874     | 0.659  | 0.751    | 0.734 | 0.944 |
| XGBoost (Ensemble)         | 0.916    | 0.794     | 0.347  | 0.483    | 0.490 | 0.850 |

## d) Key Observations for Each Machine Learning Model

| ML Model Name             | Key Observations |
|---------------------------|------------------|
| Logistic Regression       | High accuracy but extremely low recall. F1 score is low, indicating poor balance between precision and recall. Likely affected by class imbalance. |
| Decision Tree             | Strong performance across all metrics. High precision and good recall lead to a balanced F1 score. Effective in detecting positive cases. |
| k-Nearest Neighbors (kNN) | Moderate performance. Accuracy is acceptable but low recall limits its effectiveness in identifying positive cases. |
| Naive Bayes               | Low accuracy. Moderate recall but very low precision results in many false positives. Performance is unstable. |
| Random Forest (Ensemble)  | Best overall model. High accuracy, precision, recall, and F1 score indicate strong and balanced predictive power. Most reliable model. |
| XGBoost (Ensemble)        | Good accuracy and precision but low recall. F1 score is moderate, performing worse than Random Forest. |


