import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data_path = './data.csv'
data = pd.read_csv(data_path)

# Thay thế '?' với NaN và xử lý giá trị thiếu
data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)

# Chuyển đổi các cột cần thiết sang kiểu số
data['total_claim_amount'] = pd.to_numeric(data['total_claim_amount'].str.replace(',', ''), errors='coerce')
data['policy_monthly_premium'] = pd.to_numeric(data['policy_monthly_premium'].str.replace(',', ''), errors='coerce')

# Loại bỏ các hàng có giá trị NaN sau chuyển đổi
data.dropna(subset=['total_claim_amount', 'policy_monthly_premium'], inplace=True)

# Mã hóa các cột phân loại
label_enc = LabelEncoder()
data['incident_state'] = label_enc.fit_transform(data['incident_state'])
data['incident_type'] = label_enc.fit_transform(data['incident_type'])
data['fraud_reported'] = label_enc.fit_transform(data['fraud_reported'])
data['insured_sex'] = label_enc.fit_transform(data['insured_sex'])
data['policy_state'] = label_enc.fit_transform(data['policy_state'])
data['insured_occupation'] = label_enc.fit_transform(data['insured_occupation'])  # Mã hóa công việc

# Định nghĩa đặc trưng (X) và mục tiêu (y)
X = data[['insured_occupation', 'total_claim_amount', 'policy_monthly_premium', 
           'incident_state', 'incident_type', 'age', 'insured_sex', 'policy_state']]
y = data['fraud_reported']

# Kiểm tra dữ liệu đã xử lý
print(X.head())

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Tầm quan trọng của các đặc trưng
feature_importances = model.feature_importances_

# Vẽ biểu đồ tầm quan trọng của các đặc trưng
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importances)[::-1]  # Sắp xếp chỉ số theo độ quan trọng
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
