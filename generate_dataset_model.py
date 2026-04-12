import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import datetime

np.random.seed(42)
n = 2000

names = ["Rajesh Kumar", "Amit Patel", "Neha Sharma", "Ravi Singh", "Anjali Verma", "Suresh Yadav"," Priya Gupta", "Vikram Joshi", "Kavita Reddy", "Manish Tiwari", "Sunita Das", "Rohan Mehta", "Pooja Nair", "Aakash Choudhary", "Sneha Iyer", "Deepak Malhotra", "Anita Sinha", "Karan Kapoor", "Meena Kumari", "Vijay Rao"]
Locations = ["Delhi", "Mumbai", "Kolkata", "Bangalore", "Chennai", "Hyderabad","Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Surat", "Patna", "Indore", "Bhopal", "Coimbatore", "Kochi", "Visakhapatnam", "Vadodara", "Agra", "Nashik", "Meerut", "Vijayawada", "Aurangabad", "Thane", "Madurai", "Jodhpur", "Raipur", "Gwalior"]
Device = ["Android", "iOS"]
data = {
    "TransactionID": [f"TXN{str(i).zfill(6)}" for i in range(1, n+1)],
    "Date": [datetime.datetime.now() - datetime.timedelta(minutes=random.randint(1, 10000)) for _ in range(n)],
    "Amount": np.random.randint(100, 50000, n),
    "Type": np.random.choice(["P2P", "Recharge", "Bill Payment", "Shopping","Subscription Payments","QR Code Payments"], n),
    "Sender": np.random.choice(names, n),
    "Receiver": np.random.choice(names, n),
    "Location": np.random.choice(Locations, n),
    "Device": np.random.choice(Device, n),
    "FraudLabel": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "Risk": np.random.choice(["Low", "Medium", "High"], n, p=[0.7, 0.2, 0.1])
}

df = pd.DataFrame(data)
df.to_csv("upi_demo_dataset.csv", index=False)
print("✅ Dataset with Sender/Receiver saved!")

# ML Training (same as before)
df_encoded = pd.get_dummies(df.drop(columns=["TransactionID", "Date", "Sender", "Receiver", "FraudLabel"]))
X = df_encoded
y = df["FraudLabel"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "fraud_model.pkl")
print("✅ Model saved as fraud_model.pkl")
