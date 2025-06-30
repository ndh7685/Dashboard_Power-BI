import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('./data_1.csv')

# Preprocess the data
data = data.fillna(data.mean(numeric_only=True))  # Fill numerical NaNs with mean
data = data.fillna('UNKNOWN')  # Fill categorical NaNs with 'UNKNOWN'

# Encode categorical variables
label_enc = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_enc.fit_transform(data[column])

# Define features (X) and target (y)
X = data.drop(columns=['fraud_reported'])  # Drop target column
y = label_enc.fit_transform(data['fraud_reported'])  # Encode target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Random Forest model with Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1, 
                           verbose=2)

grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Best Parameters:", grid_search.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Train and evaluate Random Forest for feature importance
best_model_rf = grid_search.best_estimator_  # Get the original RandomForest model
best_model_rf.fit(X_train, y_train)

# Calculate feature importances from the original RandomForest model
feature_importances = best_model_rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance for Fraud Detection')
plt.show()

# Wrap RandomForest in OneVsRestClassifier for multi-class ROC AUC calculation
if len(np.unique(y)) > 2:  # Only wrap in OneVsRestClassifier for multi-class cases
    from sklearn.multiclass import OneVsRestClassifier
    best_model_ovr = OneVsRestClassifier(best_model_rf)
else:
    best_model_ovr = best_model_rf

# Train the wrapped model (only needed for multi-class scenario)
best_model_ovr.fit(X_train, y_train)

# Calculate ROC AUC score
try:
    if len(np.unique(y)) == 2:  # Binary classification
        roc_auc = roc_auc_score(y_test, best_model_ovr.predict_proba(X_test)[:, 1])
    else:  # Multi-class classification
        roc_auc = roc_auc_score(y_test, best_model_ovr.predict_proba(X_test), multi_class='ovr', average='weighted')
    print("ROC AUC Score:", roc_auc)
except ValueError as e:
    print("Error in calculating ROC AUC Score:", e)
    roc_auc = None  # Set ROC AUC to None if there is an error
    
# Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('confuse.png')

# --- Plot Predicted Fraud Probability Distribution ---
# Get fraud probabilities for the test set
fraud_probabilities = best_model_ovr.predict_proba(X_test)[:, 1]

# Plot the distribution of fraud probabilities
plt.figure(figsize=(10, 6))
sns.histplot(fraud_probabilities, bins=20, kde=True, color='purple')
plt.title('Predicted Fraud Probability Distribution')
plt.xlabel('Predicted Fraud Probability')
plt.ylabel('Frequency')
plt.savefig('fraud_probability_distribution.png')
plt.show()
# Ví dụ: Dự đoán cho độ tuổi và giới tính cụ thể
# Tạo từ điển mã hóa giới tính từ dữ liệu ban đầu
gender_encoding = dict(zip(data['insured_sex'].unique(), label_enc.transform(data['insured_sex'].unique())))

def predict_fraud(age, insured_sex):
    # Tạo bản ghi mẫu
    sample = X.iloc[0].copy()  # Lấy một bản ghi từ dữ liệu đã huấn luyện và điều chỉnh
    sample['age'] = age
    sample['insured_sex'] = gender_encoding.get(insured_sex)  # Mã hóa giới tính sử dụng từ điển

    # Chuẩn hóa dữ liệu
    sample = scaler.transform([sample])

    # Dự đoán xác suất gian lận
    fraud_prob = best_model_ovr.predict_proba(sample)[0][1]
    print(f"Xác suất gian lận cho độ tuổi {age} và giới tính {insured_sex}: {fraud_prob:.2f}")

# Gọi hàm với các giá trị cụ thể
predict_fraud(age=35, insured_sex='MALE')
predict_fraud(age=50, insured_sex='FEMALE')
