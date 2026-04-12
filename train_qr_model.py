# ----------------------------------------------------------
# Program: Train QR Fraud Detection Model
# ----------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ----------------------------------------------------------
# Step 1: Load Dataset
# ----------------------------------------------------------
dataset_path = "qr_transaction_dataset.csv"

if not os.path.exists(dataset_path):
    print("❌ Error: 'qr_transaction_dataset.csv' not found. Please run 'generate_qr_dataset.py' first.")
    exit()

df = pd.read_csv(dataset_path)
print(f"✅ Dataset loaded successfully: {dataset_path}")
print(f"📊 Total Records: {len(df)}")

if df.empty:
    print("⚠️ Dataset is empty. Please check your data generation script.")
    exit()

print(df.head())

# ----------------------------------------------------------
# Step 2: Preprocess Data
# ----------------------------------------------------------
# Handle missing values safely
df = df.dropna(subset=["UPI_ID", "Amount", "Location", "Merchant", "Fraudulent"], how="any")

# Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df[["UPI_ID", "Amount", "Location", "Merchant"]])

# Target column (convert 'Yes'/'No' → 1/0)
y = df["Fraudulent"].map({"Yes": 1, "No": 0})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# Step 3: Train Model
# ----------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# ----------------------------------------------------------
# Step 4: Evaluate
# ----------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {round(accuracy * 100, 2)}%")
print("\n📘 Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# Step 5: Save Model
# ----------------------------------------------------------
model_file = "qr_model.pkl"
joblib.dump(model, model_file)
print(f"\n✅ Model saved as: {model_file}")

# ----------------------------------------------------------
# Step 6: Save Model Accuracy for Dashboard
# ----------------------------------------------------------
acc_file = "qr_model_accuracy.txt"
with open(acc_file, "w") as f:
    f.write(str(round(accuracy * 100, 2)))

print(f"📄 Model accuracy saved in: {acc_file}")

# ----------------------------------------------------------
# Step 7: Summary
# ----------------------------------------------------------
print("\n✅ Training complete!")
print(f"👉 Model File: {model_file}")
print(f"👉 Accuracy File: {acc_file}")
print(f"👉 Accuracy: {round(accuracy * 100, 2)}%")
