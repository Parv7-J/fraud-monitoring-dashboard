import numpy as np
import pandas as pd
df=pd.read_csv("/content/transactions_train.csv")
df.head()
legit=df[df.is_fraud==0]
fraud=df[df.is_fraud==1]
print(legit.shape)
print(fraud.shape)
legit.transaction_amount.describe()
fraud.transaction_amount.describe()
# Check mean values for each group (fraud vs. non-fraud)
df.groupby('is_fraud').mean(numeric_only=True)
legit_sample=legit.sample(n=1010)
dfnew=pd.concat([legit_sample,fraud],axis=0)
dfnew.head()
dfnew.tail()
# Check mean values for each group (fraud vs. non-fraud)
dfnew.groupby('is_fraud').mean(numeric_only=True)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import numpy as np

# Check for existing columns before dropping
print(dfnew.columns)  # Print the existing columns in your DataFrame

# Drop only existing columns
columns_to_drop = ["payer_email_anonymous", "payee_ip_anonymous", "transaction_date"]
existing_columns = [col for col in columns_to_drop if col in dfnew.columns]
dfnew = dfnew.drop(columns=existing_columns)

# Encode transaction_channel: 1 for 'w', 0 for 'mobile'
dfnew["transaction_channel"] = dfnew["transaction_channel"].map({"w": 1, "mobile": 0})

# Encode anonymized columns
dfnew["payer_mobile_anonymous"] = dfnew["payer_mobile_anonymous"].astype('category').cat.codes
dfnew["transaction_id_anonymous"] = dfnew["transaction_id_anonymous"].astype('category').cat.codes
dfnew["payee_id_anonymous"] = dfnew["payee_id_anonymous"].astype('category').cat.codes

# Split dataset into features and target
X = dfnew.drop(columns=["is_fraud"])
y = dfnew["is_fraud"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ... (rest of your code)



# Initialize XGBoost model
# Initialize XGBoost model with improved hyperparameters
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    use_label_encoder=False,
    n_estimators=500,  # More estimators (with early stopping to prevent overfitting)
    learning_rate=0.03,  # Lower learning rate for better accuracy
    max_depth=3,  # Reduce complexity (prevents overfitting)
    min_child_weight=5,  # Require more samples per split
    gamma=10,  # Higher value to avoid unnecessary splits
    subsample=0.9,  # Uses more data per tree
    colsample_bytree=0.9,  # Uses more features per tree
    reg_alpha=0.1,  # L1 regularization (unchanged)
    reg_lambda=2.0  # Stronger L2 regularization (reduces overfitting)
)

# Train the model with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
     # Stops training if no improvement in 30 rounds
    verbose=True
)


# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
print("Best Precision-Recall Tradeoff:", max(2 * (precision * recall) / (precision + recall)))
import joblib
joblib.dump(model,'xgboost_fraud_model.pk1')
# Print predicted fraud labels (0 = not fraud, 1 = fraud)
print("Predicted Fraud Labels:")
print(y_pred)

# Print fraud probability scores (higher score = higher fraud risk)
print("Fraud Probability Scores:")
print(y_pred_proba)
import pandas as pd

# Create DataFrame with actual labels, predicted labels, and fraud scores
results_df = pd.DataFrame({
    "Actual Label": y_test.values,
    "Predicted Label": y_pred,
    "Fraud Probability Score": y_pred_proba
})

# Display first few rows
print(results_df.head(10))
avg_fraud_prob = y_pred_proba.mean()
print("Average Fraud Probability Score:", avg_fraud_prob)
