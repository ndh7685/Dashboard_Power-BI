import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load data
data = pd.read_csv('./data.csv')

# Print columns to check for 'policy_monthly_premium'
print(data.columns)

# Optional: Clean column names
data.columns = data.columns.str.strip().str.lower()  # Remove whitespaces and lowercase

# Clean 'policy_monthly_premium' by removing commas and converting to numeric
if 'policy_monthly_premium' in data.columns:
    data['policy_monthly_premium'] = pd.to_numeric(data['policy_monthly_premium'].str.replace(',', ''))
else:
    print("Column 'policy_monthly_premium' does not exist.")

# Handle missing values if any
data = data.replace('?', None)

# Convert categorical columns to numeric
label_enc = LabelEncoder()
data['incident_state'] = label_enc.fit_transform(data['incident_state'])
data['incident_type'] = label_enc.fit_transform(data['incident_type'])
data['fraud_reported'] = label_enc.fit_transform(data['fraud_reported'])

# Define features (X) and target (y)
X = data[['insured_occupation', 'total_claim_amount', 'policy_annual_premium', 'incident_state', 'incident_type', 'age']]
y = data['fraud_reported']

# Remaining code stays the same...


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# View feature importance
feature_importances = model.feature_importances_
for feature, importance in zip(X.columns, feature_importances):
    print(f'{feature}: {importance:.4f}')
