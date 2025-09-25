import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

# 1. Load the engineered dataset
df = pd.read_csv('archive/Mental_Health_Lifestyle_Dataset_with_features.csv')

# 2. Create binary target: High_Stress (1 if Stress Level is 'High', else 0)
df['High_Stress'] = (df['Stress Level'].str.lower() == 'high').astype(int)

# 3. Select features (drop columns not used for prediction)
features = [
    'Age', 'Gender', 'Country', 'Exercise Level', 'Diet Type', 'Sleep Hours',
    'Work Hours per Week', 'Screen Time per Day (Hours)', 'Social Interaction Score',
    'Happiness Score', 'Age Group'
]
X = df[features]
y = df['High_Stress']

# 4. Encode categorical variables
categorical_cols = ['Gender', 'Country', 'Exercise Level', 'Diet Type', 'Age Group']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train baseline model (Logistic Regression)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# 7. Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 8. Evaluate models
def print_metrics(y_true, y_pred, y_prob, model_name):
    print(f'\n--- {model_name} ---')
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Precision:', precision_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred))
    print('F1 Score:', f1_score(y_true, y_pred))
    print('ROC AUC:', roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred))

print_metrics(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression')
print_metrics(y_test, y_pred_rf, y_prob_rf, 'Random Forest')

# 9. Save best model (Random Forest) and predictions
joblib.dump(rf, 'archive/high_stress_rf_model.joblib')
predictions = X_test.copy()
predictions['High_Stress_Actual'] = y_test.values
predictions['High_Stress_Predicted'] = y_pred_rf
predictions['High_Stress_Probability'] = y_prob_rf
predictions.to_csv('archive/high_stress_predictions.csv', index=False)

print('\nModel and predictions saved to archive/.')
