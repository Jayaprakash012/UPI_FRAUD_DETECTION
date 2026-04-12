# ==================================================
# Program: QR + Transaction Dataset Generator + ML Model Trainer
# ==================================================

import os
import csv
import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import qrcode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ==================================================
# CONFIG
# ==================================================

TOTAL_MERCHANTS = 40
TRANSACTIONS_PER_MERCHANT = 20

CSV_FILE = "qr_transaction_dataset.csv"
QR_FOLDER = "qr_codes"
MODEL_FILE = "qr_model.pkl"
ACCURACY_FILE = "qr_model_accuracy.txt"

os.makedirs(QR_FOLDER, exist_ok=True)

fake = Faker()

merchant_categories = [
    "Grocery", "Mobile Shop", "Restaurant",
    "Electronics", "Clothing", "Cafe",
    "Pharmacy", "Book Store", "Supermarket"
]

# ==================================================
# DATASET STORAGE
# ==================================================

dataset = []
txn_no = 1

# ==================================================
# DATA GENERATION LOOP
# ==================================================

for m in range(TOTAL_MERCHANTS):

    merchant_id = f"MID{m+1:04}"
    merchant_name = fake.company()
    merchant_city = fake.city()
    merchant_category = random.choice(merchant_categories)
    merchant_upi = merchant_name.replace(" ", "").lower()[:7] + "@upi"

    for _ in range(TRANSACTIONS_PER_MERCHANT):

        transaction_id = f"TXN{txn_no:05}"
        txn_no += 1

        customer_name = fake.name()
        customer_phone = fake.msisdn()[:10]
        amount = round(random.uniform(50, 8000), 2)
        payment_status = random.choice(["SUCCESS", "FAILED"])
        time = (datetime.now() - timedelta(minutes=random.randint(0, 10000))).strftime("%Y-%m-%d %H:%M:%S")

        # ------------------------------
        # FRAUD LOGIC
        # ------------------------------
        fraudulent = "Yes" if amount > 3000 or random.random() < 0.10 else "No"

        # ------------------------------
        # QR DATA CONTENT
        # ------------------------------
        qr_data = f"""
MERCHANT TRANSACTION RECEIPT
--------------------------------------
MERCHANT_ID:{merchant_id}
MERCHANT_NAME:{merchant_name}
CITY:{merchant_city}
CATEGORY:{merchant_category}
MERCHANT_UPI:{merchant_upi}

TRANSACTION_ID:{transaction_id}
CUSTOMER:{customer_name}
PHONE:{customer_phone}
AMOUNT:{amount}
STATUS:{payment_status}

FRAUD:{fraudulent}
TIME:{time}
"""

        # ------------------------------
        # SAVE QR IMAGE
        # ------------------------------
        qr_path = os.path.join(QR_FOLDER, f"{transaction_id}.png")
        qr_img = qrcode.make(qr_data)
        with open(qr_path, "wb") as f:
            qr_img.save(f)

        # ------------------------------
        # SAVE ROW TO DATASET
        # ------------------------------
        dataset.append({
            "Merchant_ID": merchant_id,
            "Merchant_Name": merchant_name,
            "Merchant_Category": merchant_category,
            "Merchant_City": merchant_city,
            "Merchant_UPI": merchant_upi,
            "Transaction_ID": transaction_id,
            "Customer_Name": customer_name,
            "Customer_Phone": customer_phone,
            "Amount": amount,
            "Payment_Status": payment_status,
            "Fraudulent": fraudulent,
            "QR_Image_Path": qr_path,
            "Date_Time": time
        })

# ==================================================
# SAVE CSV
# ==================================================

df = pd.DataFrame(dataset)
df.to_csv(CSV_FILE, index=False)

# ==================================================
# MACHINE LEARNING MODEL TRAINING
# ==================================================

print("\n📊 Training Fraud Detection Model...")

# Convert Fraudulent column to binary
df["Fraudulent"] = df["Fraudulent"].map({"Yes": 1, "No": 0})

X = df[["Amount"]]
y = df["Fraudulent"]

# ------------------------------
# TRAIN / TEST SPLIT
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ------------------------------
# TRAIN MODEL
# ------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# TEST MODEL
# ------------------------------
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# ------------------------------
# SAVE MODEL + ACCURACY
# ------------------------------
joblib.dump(model, MODEL_FILE)

with open(ACCURACY_FILE, "w") as f:
    f.write(str(accuracy))

# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n==================== OUTPUT ====================")
print("✅ Dataset & QR Generation Completed!")
print("📁 CSV File            :", CSV_FILE)
print("🧾 QR Images Folder    :", QR_FOLDER)
print("📊 Total Records       :", len(dataset))

print("\n🤖 ML Model Training Completed!")
print("📦 Model Saved As      :", MODEL_FILE)
print("🎯 Model Accuracy      :", accuracy, "%")
print("📝 Accuracy Text File  :", ACCURACY_FILE)
print("================================================")
