import flask
from flask import jsonify, render_template, request
import pandas as pd
import joblib
import random
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import numpy as np
import re
# using OpenCV's QRCodeDetector instead of pyzbar to avoid external DLL dependencies

def decode(pil_image):
    """Decode QR code from a PIL image using OpenCV's QRCodeDetector."""
    # Convert PIL image to OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(cv_img)
    if data:
        class Decoded:
            def __init__(self, data):
                self.data = data.encode("utf-8")
        return [Decoded(data)]
    return []

app = flask.Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPI_CSV = os.path.join(BASE_DIR, "upi_demo_dataset.csv")
QR_CSV = os.path.join(BASE_DIR, "qr_transaction_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
QR_MODEL_PATH = os.path.join(BASE_DIR, "qr_model.pkl")
RECORDS_DIR = os.path.join(BASE_DIR, "records")
os.makedirs(RECORDS_DIR, exist_ok=True)

APPROVED_CSV = os.path.join(RECORDS_DIR, "approved_transactions.csv")
FRAUD_CSV = os.path.join(RECORDS_DIR, "fraudulent_transactions.csv")
REVIEW_CSV = os.path.join(RECORDS_DIR, "review_transactions.csv")
def safe_load_model(path):
    """Attempt to load a joblib model. Return None if unavailable or on error."""
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Failed loading model {path}: {e}")
        return None


def safe_read_csv(path, default_columns=None):
    """Attempt to read a CSV and return a DataFrame. If missing or unreadable, return
    an empty DataFrame with default_columns (if provided).
    """
    if not os.path.exists(path):
        print(f"CSV not found: {path}")
        if default_columns:
            return pd.DataFrame(columns=default_columns)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed reading CSV {path}: {e}")
        if default_columns:
            return pd.DataFrame(columns=default_columns)
        return pd.DataFrame()


# Load model and datasets safely so the app doesn't crash at import time
model = safe_load_model(MODEL_PATH)
qr_model = safe_load_model(QR_MODEL_PATH)

# expected columns used throughout the app (best-effort)
expected_upi_cols = [
    "TransactionID",
    "Date",
    "Sender",
    "Receiver",
    "Amount",
    "Type",
    "Location",
    "Device",
    "FraudLabel",
]

df = safe_read_csv(UPI_CSV, default_columns=expected_upi_cols)
qr_dataset = safe_read_csv(QR_CSV)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
if 'Date' in qr_dataset.columns:
    qr_dataset['Date'] = pd.to_datetime(qr_dataset['Date'], errors='coerce')



# ==== HELPER FUNCTIONS ====
def ensure_csv(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def compute_feature_scores(txn, df):
    sender = txn.get('sender_id')
    amount = float(txn.get('amount', 0))
    device = txn.get('device_id')
    ts = txn.get('timestamp')
    location = txn.get('location')

    try:
        txn_dt = pd.to_datetime(ts)
    except Exception:
        txn_dt = None

    # --- Unknown Device ---
    sender_devices = df[df['Sender'] == sender]['DeviceID'].dropna().unique().tolist() if 'DeviceID' in df.columns else []
    if not sender_devices:
        unknown_device_score = 20 if device else 0
    elif device not in sender_devices:
        unknown_device_score = min(95, 40 + len(sender_devices) * 20)
    else:
        unknown_device_score = min(80, len(sender_devices) * 20)

    # --- Location Consistency ---
    sender_locations = df[df['Sender'] == sender]['Location'].dropna().unique().tolist() if 'Location' in df.columns else []
    if len(sender_locations) <= 1:
        location_consistency_score = 10
    elif location not in sender_locations:
        location_consistency_score = 90
    else:
        location_consistency_score = 40

    # --- Unusual Amount ---
    sender_amounts = df[df['Sender'] == sender]['Amount'].dropna().astype(float) if 'Amount' in df.columns else pd.Series(dtype=float)
    avg_amount = sender_amounts.mean() if not sender_amounts.empty else 0
    unusual_amount_score = 80 if amount >= 30000 or (avg_amount > 0 and amount > 5 * avg_amount) else 10

    # --- Time Anomaly ---
    time_anomaly_score = 0
    if txn_dt is not None:
        h = txn_dt.hour
        hours_hist = df[df['Sender'] == sender]['Date'].dt.hour if 'Date' in df.columns else pd.Series()
        if not hours_hist.empty:
            hour_freq = (hours_hist == h).sum()
            if len(hours_hist) > 0 and (hour_freq / len(hours_hist)) < 0.05:
                time_anomaly_score = 40
        if 12 <= h < 18 and amount >= 30000:
            time_anomaly_score = 85

    # --- Weighted Probability ---
    weights = {'unusual_amount': 0.35, 'unknown_device': 0.25, 'time_anomaly': 0.2, 'location_consistency': 0.2}
    fraud_prob = (
        unusual_amount_score * weights['unusual_amount'] +
        unknown_device_score * weights['unknown_device'] +
        time_anomaly_score * weights['time_anomaly'] +
        location_consistency_score * weights['location_consistency']
    ) / 100

    return {
        "unusual_amount": unusual_amount_score,
        "unknown_device": unknown_device_score,
        "time_anomaly": time_anomaly_score,
        "location_consistency": location_consistency_score,
        "fraud_probability": round(fraud_prob, 3),
        "avg_amount": float(avg_amount)
    }

# ==== ROUTES ====

@app.route("/evaluate", methods=["POST"])
def evaluate_transaction():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No transaction data"}), 400

    features = compute_feature_scores(data, df)
    radar_data = {
        "labels": [
            "Unknown Device",
            "Location Consistency",
            "Unusual Amount",
            "Time Anomaly"
        ],
        "scores": [
            features["unknown_device"],
            features["location_consistency"],
            features["unusual_amount"],
            features["time_anomaly"]
        ]
    }

    return jsonify({
        "features": features,
        "radar": radar_data
    })

# Duplicate imports and path setup removed (already defined above)

@app.route('/decision', methods=['POST'])
def handle_decision():
    try:
        data = request.get_json()
        decision = data.get('decision')
        txn = data.get('txn')

        if not txn or not decision:
            return jsonify({"status": "error", "message": "Missing data"}), 400

        # map decision to correct filename
        decision_map = {
            'approve': 'approved_transactions.csv',
            'flag': 'fraudulent_transactions.csv',
            'review': 'review_transactions.csv'
        }

        filename = decision_map.get(decision.lower())
        if not filename:
            return jsonify({"status": "error", "message": "Invalid decision"}), 400

        file_path = os.path.join(RECORDS_DIR, filename)

        # convert transaction to DataFrame
        df_new = pd.DataFrame([txn])

        # append or create file
        if os.path.exists(file_path):
            df_old = pd.read_csv(file_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(file_path, index=False)

        return jsonify({
            "status": "success",
            "message": f"Transaction saved to {filename}"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")

@app.route("/transaction")
def get_transaction():
    s = df.sample(1).iloc[0].to_dict()
    parties = f"{s['Sender']} → {s['Receiver']}"
    # If model not available, return sample transaction with no risk prediction
    if model is None:
        output = {
            "TransactionID": s.get("TransactionID"),
            "Date": s.get("Date"),
            "Amount": f"₹{s.get('Amount')}",
            "Type": s.get("Type"),
            "Parties": parties,
            "RiskScore": None,
            "Status": "Model unavailable",
            "Anomalies": "Unknown"
        }
        return jsonify(output)

    features = pd.get_dummies(pd.DataFrame([s]).drop(columns=["TransactionID", "Date", "Sender", "Receiver", "FraudLabel"]))
    for col in getattr(model, "feature_names_in_", []):
        if col not in features.columns:
            features[col] = 0
    features = features[getattr(model, "feature_names_in_", features.columns)]

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    risk_score = round(prob * 100, 2)
    status = "High Risk" if risk_score > 60 else "Low Risk"

    output = {
        "TransactionID": s["TransactionID"],
        "Date": s["Date"],
        "Amount": f"₹{s['Amount']}",
        "Type": s["Type"],
        "Parties": parties,
        "RiskScore": risk_score,
        "Status": status,
        "Anomalies": "Unusual transaction amount" if pred == 1 else "Completed"
    }

    return jsonify(output)

@app.route("/search_data")
def search_data():
    query = request.args.get("q", "").lower()
    filtered_df = df[
        df['TransactionID'].str.lower().str.contains(query) |
        df['Sender'].str.lower().str.contains(query) |
        df['Receiver'].str.lower().str.contains(query) |
        df['Type'].str.lower().str.contains(query)
    ]
    samples = filtered_df.to_dict(orient="records")
    output = []

    for s in samples:
        parties = f"{s['Sender']} → {s['Receiver']}"
        # If model missing, skip prediction and return neutral values
        if model is None:
            output.append({
                "TransactionID": s.get("TransactionID"),
                "Date": s.get("Date"),
                "Amount": f"₹{s.get('Amount')}",
                "Type": s.get("Type"),
                "Parties": parties,
                "RiskScore": None,
                "Status": "Model unavailable",
                "Anomalies": "Unknown"
            })
            continue

        features = pd.get_dummies(pd.DataFrame([s]).drop(columns=["TransactionID", "Date", "Sender", "Receiver", "FraudLabel"]))
        for col in getattr(model, "feature_names_in_", []):
            if col not in features.columns:
                features[col] = 0
        features = features[getattr(model, "feature_names_in_", features.columns)]

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        risk_score = round(prob * 100, 2)
        status = "High Risk" if risk_score > 60 else "Low Risk"

        output.append({
            "TransactionID": s["TransactionID"],
            "Date": s["Date"],
            "Amount": f"₹{s['Amount']}",
            "Type": s["Type"],
            "Parties": parties,
            "RiskScore": risk_score,
            "Status": status,
            "Anomalies": "Unusual transaction amount" if pred == 1 else "Completed"
        })
    return jsonify(output)

@app.route("/get_chart_data")
def get_chart_data():
    transaction_counts = df['Type'].value_counts().to_dict()
    return jsonify(transaction_counts)

@app.route("/analyze_qr", methods=["POST"])
def analyze_qr():
    data = request.get_json()
    payee_vpa = data.get("payee_vpa")
    merchant_code = data.get("merchant_code")

    result = qr_dataset[
        (qr_dataset["payee_vpa"] == payee_vpa) &
        (qr_dataset["merchant_code"] == merchant_code)
    ]

    if not result.empty:
        row = result.iloc[0]
        return jsonify({
            "payee_vpa": row["payee_vpa"],
            "merchant_code": row["merchant_code"],
            "location": row["location"],
            "device": row["device"],
            "transaction_type": row["transaction_type"]
        })
    else:
        return jsonify({"error": "No matching record found in dataset"})

@app.route("/analyze_transaction", methods=["POST"])
def analyze_transaction():
    tx_id = request.form.get("transaction_id", "").strip()
    if tx_id == "":
        return render_template("analysis.html", error="Please enter a Transaction ID")

    result = df[df["TransactionID"].astype(str) == tx_id]
    if result.empty:
        return render_template("analysis.html", error="Transaction not Found")

    row = result.iloc[0].to_dict()

    # Transaction Type Bar Chart
    transaction_counts = df['Type'].value_counts().to_dict()
    chart_labels = list(transaction_counts.keys())
    chart_data = list(transaction_counts.values())

    # Risk Pie Chart (FIXED)
    if 'FraudLabel' in df.columns:
        risk_counts = df['FraudLabel'].value_counts().to_dict()
    else:
        risk_counts = {"Unknown": 0}
    risk_labels = list(risk_counts.keys())
    risk_data = list(risk_counts.values())

    # Calculate RiskScore (if model available)
    if model is None:
        risk_score = None
    else:
        features = pd.get_dummies(pd.DataFrame([row]).drop(columns=["TransactionID", "Date", "Sender", "Receiver", "FraudLabel"], errors="ignore"))
        for col in getattr(model, "feature_names_in_", []):
            if col not in features.columns:
                features[col] = 0
        features = features[getattr(model, "feature_names_in_", features.columns)]
        prob = model.predict_proba(features)[0][1]
        risk_score = round(prob * 100, 2)

    transaction = {
        "TransactionID": row["TransactionID"],
        "Date": row["Date"],
        "Sender": row["Sender"],
        "Receiver": row["Receiver"],
        "Type": row["Type"],
        "Amount": row["Amount"],
        "Location": row["Location"],
        "Device": row["Device"],
        "RiskScore": risk_score,
        "FraudLabel": row["FraudLabel"]
    }

    # Behavior Profile for radar chart
    behavior_labels = ["Amount", "RiskScore", "FraudLabel", "LocationCount", "DailyCount"]
    max_amount = df['Amount'].max()
    amount_norm = transaction["Amount"] / max_amount
    fraud = transaction["FraudLabel"]
    location_count = len(df[df["Location"] == transaction["Location"]]) / df.shape[0]
    daily_count = len(df[df["Date"].dt.date == pd.to_datetime(transaction["Date"]).date()]) / df.shape[0]
    behavior_data = [round(amount_norm, 2), round(transaction["RiskScore"]/100, 2), fraud, round(location_count,2), round(daily_count,2)]

    return render_template(
        "analysis.html",
        transaction=transaction,
        chart_labels=chart_labels,
        chart_data=chart_data,
        risk_labels=risk_labels,
        risk_data=risk_data,
        behavior_labels=behavior_labels,
        behavior_data=behavior_data
    )
@app.route("/qranalysis")
def qr_upload():
    """Render QR upload test page with dynamic model and dataset info"""
    dataset_path = QR_CSV
    model_path = QR_MODEL_PATH

    # Dataset status
    dataset_status = "❌ Not Found"
    dataset_rows = 0
    if os.path.exists(dataset_path):
        dataset_status = f"✅ Dataset loaded successfully: {os.path.basename(dataset_path)}"
        try:
            dataset_rows = len(pd.read_csv(dataset_path))
        except:
            dataset_rows = 0

    # Model status
    model_status = "❌ Model not found"
    model_accuracy = "Unknown"
    if os.path.exists(model_path):
        model_status = f"✅ Model loaded successfully:"
        acc_file = os.path.join(BASE_DIR, "qr_model_accuracy.txt")
        if os.path.exists(acc_file):
            with open(acc_file, "r") as f:
                model_accuracy = f.read().strip()

    return render_template(
        "qr_upload.html",
        dataset_status=dataset_status,
        dataset_rows=dataset_rows,
        model_status=model_status,
        model_accuracy=model_accuracy
    )
def parse_qr_fields(qr_text: str) -> dict:
    """
    Parse KEY:VALUE fields from your receipt-like QR text.
    Example:
      MERCHANT_ID:MID0001 MERCHANT_NAME:Kelly, Harper and Chen CITY:South Janiceview ...
    Returns dict: {"MERCHANT_ID": "...", "MERCHANT_NAME": "...", ...}
    """
    fields = {}
    if not qr_text:
        return fields

    pattern = r"([A-Z_]+):"
    matches = list(re.finditer(pattern, qr_text))
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(qr_text)
        value = qr_text[start:end].strip()
        fields[key] = value
    return fields
def compute_merchant_behavior(merchant_rows: pd.DataFrame,
                              global_df: pd.DataFrame,
                              current_row: pd.Series) -> dict:
    """
    Build behavior profile for a merchant, returning:
      {
        "labels": [...],
        "current": [...],   # 0–100
        "baseline": [...]   # 0–100
      }
    Axes:
      - Amount Pattern
      - Time Frequency
      - Location Consistency
      - Merchant Trust
      - Device Pattern
      - Transaction Velocity
    """

    labels = [
        "Amount Pattern",
        "Time Frequency",
        "Location Consistency",
        "Merchant Trust",
        "Device Pattern",
        "Transaction Velocity"
    ]

    def num_series(df, col):
        if col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    current = [50] * len(labels)
    baseline = [50] * len(labels)

    if merchant_rows is None or merchant_rows.empty:
        return {"labels": labels, "current": current, "baseline": baseline}

    # dates
    if "Date_Time" in merchant_rows.columns:
        m_dates = pd.to_datetime(merchant_rows["Date_Time"], errors="coerce")
    else:
        m_dates = pd.Series(dtype="datetime64[ns]")

    if "Date_Time" in global_df.columns:
        g_dates = pd.to_datetime(global_df["Date_Time"], errors="coerce")
    else:
        g_dates = pd.Series(dtype="datetime64[ns]")

    # ---------- 1. Amount Pattern ----------
    m_amounts = num_series(merchant_rows, "Amount").dropna()
    g_amounts = num_series(global_df, "Amount").dropna()
    try:
        cur_amt = float(current_row.get("Amount"))
    except Exception:
        cur_amt = None

    if cur_amt is not None and not m_amounts.empty:
        mean_m = m_amounts.mean()
        if mean_m > 0:
            ratio = abs(cur_amt - mean_m) / mean_m
            if ratio <= 0.25:
                s = 85
            elif ratio <= 1:
                s = 60
            else:
                s = 35
        else:
            s = 50
        current[0] = s
    else:
        current[0] = 50

    if not g_amounts.empty:
        mean_g = g_amounts.mean()
        if mean_g > 0 and not m_amounts.empty:
            ratio_bg = abs(m_amounts.mean() - mean_g) / mean_g
            baseline[0] = max(20, 80 - ratio_bg * 40)
        else:
            baseline[0] = 60

    # ---------- 2. Time Frequency ----------
    time_score = 50
    baseline_time = 50
    mhours = m_dates.dt.hour.dropna() if not m_dates.empty else pd.Series(dtype="int64")

    try:
        cur_dt = pd.to_datetime(current_row.get("Date_Time"), errors="coerce")
    except Exception:
        cur_dt = None

    if cur_dt is not None and not mhours.empty:
        cur_hour = cur_dt.hour
        freq = (mhours == cur_hour).sum()
        prop = freq / max(1, len(mhours))
        if prop >= 0.25:
            time_score = 80
        elif prop >= 0.10:
            time_score = 60
        else:
            time_score = 35
    current[1] = time_score

    if not g_dates.empty and not mhours.empty:
        ghours = g_dates.dt.hour.dropna()
        if not ghours.empty:
            top_hour = mhours.mode().iloc[0]
            g_prop = (ghours == top_hour).sum() / max(1, len(ghours))
            baseline_time = 60 + (g_prop - 0.1) * 80
            baseline_time = max(20, min(80, baseline_time))
    baseline[1] = baseline_time

    # ---------- 3. Location Consistency ----------
    loc_col = "Merchant_City" if "Merchant_City" in merchant_rows.columns else "Location"
    m_locs = merchant_rows[loc_col].dropna().astype(str) if loc_col in merchant_rows.columns else pd.Series(dtype=str)
    if not m_locs.empty:
        unique_count = m_locs.nunique()
        if unique_count == 1:
            current[2] = 85
        elif unique_count <= 3:
            current[2] = 65
        else:
            current[2] = 40
    else:
        current[2] = 50

    if loc_col in global_df.columns:
        g_locs = global_df[loc_col].dropna().astype(str)
        baseline[2] = 60 if not g_locs.empty else 50

    # ---------- 4. Merchant Trust ----------
    if "Fraudulent" in merchant_rows.columns:
        fraud_vals = merchant_rows["Fraudulent"].map({"Yes": 1, "No": 0}).fillna(0)
        fraud_rate = fraud_vals.mean()
        current[3] = max(10, 100 - fraud_rate * 100)
    else:
        current[3] = 60

    if "Fraudulent" in global_df.columns:
        g_fraud = global_df["Fraudulent"].map({"Yes": 1, "No": 0}).fillna(0)
        baseline[3] = max(10, 100 - g_fraud.mean() * 100)
    else:
        baseline[3] = 60

    # ---------- 5. Device Pattern ----------
    current[4] = 60
    baseline[4] = 55

    # ---------- 6. Transaction Velocity ----------
    vel_score = 50
    base_vel = 50
    if not m_dates.empty:
        m_dates_only = m_dates.dt.date.dropna()
        tx_per_day = m_dates_only.value_counts().mean()
        if tx_per_day >= 20:
            vel_score = 80
        elif tx_per_day >= 10:
            vel_score = 65
        elif tx_per_day >= 3:
            vel_score = 55
        else:
            vel_score = 40
    current[5] = vel_score

    if not g_dates.empty:
        g_dates_only = g_dates.dt.date.dropna()
        g_tx_per_day = g_dates_only.value_counts().mean()
        if g_tx_per_day:
            base_vel = 55
    baseline[5] = base_vel

    return {
        "labels": labels,
        "current": [round(x) for x in current],
        "baseline": [round(x) for x in baseline]
    }

@app.route("/upload_qr", methods=["POST"])
def upload_qr():
    try:
        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")
        decoded = decode(image)

        if not decoded:
            return jsonify({"error": "Invalid QR code"})

        qr_raw = decoded[0].data.decode("utf-8")
        print("QR RAW:", qr_raw)

        # Parse KEY:VALUE pairs from the receipt text
        kv = parse_qr_fields(qr_raw)

        # ---------- extract UPI from UPI URL if present ----------
        upi = None
        upi_match = re.search(r"pa=([^&]+)", qr_raw)
        if upi_match:
            upi = upi_match.group(1).strip().lower()
            print("EXTRACTED UPI:", upi)

        attributes = {}
        fraud_status = "Unknown"

        # neutral behavior by default (in case no match)
        behavior_profile = compute_merchant_behavior(
            merchant_rows=pd.DataFrame(columns=qr_dataset.columns),
            global_df=qr_dataset,
            current_row=pd.Series(dtype=object),
        )

        # =====================================================
        # 1) TRY MATCHING YOUR CSV ON Merchant_UPI
        #    AND COMPUTE TOTAL AMOUNT RECEIVED BY THIS MERCHANT
        # =====================================================
        if upi and "Merchant_UPI" in qr_dataset.columns:
            qr_dataset["Merchant_UPI"] = (
                qr_dataset["Merchant_UPI"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            merchant_rows = qr_dataset[qr_dataset["Merchant_UPI"] == upi]

            if not merchant_rows.empty:
                row = merchant_rows.iloc[0]
                print("DATASET MATCH:", row.to_dict())

                total_amount = (
                    pd.to_numeric(merchant_rows["Amount"], errors="coerce")
                    .fillna(0)
                    .sum()
                )

                attributes = {
                    "amount": float(total_amount),   # TOTAL to merchant
                    "payee": row.get("Merchant_Name"),
                    "merchant": row.get("Merchant_Name"),
                    "vpa": row.get("Merchant_UPI"),
                    "payee_vpa": row.get("Merchant_UPI"),
                    "location": row.get("Merchant_City"),
                    "category": row.get("Merchant_Category"),
                    "date_time": row.get("Date_Time"),
                }

                fraud_status = row.get("Fraudulent", "Unknown")

                # compute real behavior profile for this merchant
                behavior_profile = compute_merchant_behavior(
                    merchant_rows=merchant_rows,
                    global_df=qr_dataset,
                    current_row=row,
                )

        # =====================================================
        # 2) FALLBACK: IF NO DATASET MATCH, USE QR TEXT FIELDS
        # =====================================================
        if not attributes:
            print("FALLBACK: USING QR FIELDS ONLY")

            amt_str = kv.get("AMOUNT")
            try:
                amount_val = float(amt_str) if amt_str else None
            except Exception:
                amount_val = None

            merchant_upi = kv.get("MERCHANT_UPI")
            if merchant_upi:
                merchant_upi = merchant_upi.strip()

            attributes = {
                "amount": amount_val,  # only single txn, no history here
                "payee": kv.get("MERCHANT_NAME"),
                "merchant": kv.get("MERCHANT_NAME"),
                "vpa": merchant_upi or (upi if upi else None),
                "payee_vpa": merchant_upi or (upi if upi else None),
                "location": kv.get("CITY"),
                "category": kv.get("CATEGORY"),
                "date_time": kv.get("TIME"),
            }

            fraud_status = kv.get("FRAUD", fraud_status)
            # behavior_profile stays as neutral (already set above)

        print("ATTRIBUTES SENT:", attributes)
        print("FRAUD STATUS:", fraud_status)
        print("BEHAVIOR PROFILE:", behavior_profile)

        return jsonify({
            "qr_valid": True,
            "qr_data": qr_raw,
            "fraud": fraud_status,
            "attributes": attributes,
            "behavior": behavior_profile
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)