"""
LedgerGuard Synthetic Data Generator
Generates realistic SMB bookkeeping & expense transaction streams
with subtle, realistic operational anomalies (typos, duplicates, velocity bursts, off-hour spikes).
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

VENDORS = {
    "AWS Cloud Services": {"category": "Hosting & Cloud", "mean": 7500, "std": 1200},
    "Google Workspace": {"category": "Hosting & Cloud", "mean": 3200, "std": 400},
    "Office Depot": {"category": "Office Supplies", "mean": 1800, "std": 600},
    "Local Stationers": {"category": "Office Supplies", "mean": 650, "std": 200},
    "Uber Corporate": {"category": "Travel & Commute", "mean": 600, "std": 250},
    "BlueDart Logistics": {"category": "Logistics & Shipping", "mean": 2400, "std": 800},
    "Swiggy for Work": {"category": "Food & Meals", "mean": 850, "std": 300},
    "WeWork Coworking": {"category": "Rent & Space", "mean": 30000, "std": 2500},
    "Freelance Developer": {"category": "Contractors", "mean": 35000, "std": 6000},
    "City Electricity Board": {"category": "Utilities", "mean": 4500, "std": 900},
}

PLATFORMS = ["android", "ios", "web"]
PAYMENT_METHODS = ["UPI", "Corporate Card", "Net Banking", "Cash"]
USERS = [f"USR_{i:03d}" for i in range(101, 126)]  # 25 SMB accounts


def generate_transactions(num_records: int = 5000, anomaly_rate: float = 0.05) -> pd.DataFrame:
    records = []
    base_time = datetime(2026, 7, 1, 9, 0, 0)
    total_anomalies = int(num_records * anomaly_rate)
    anomaly_indices = set(random.sample(range(num_records), total_anomalies))

    for i in range(num_records):
        tx_id = f"TX_{100000 + i}"
        user_id = random.choice(USERS)
        vendor_name = random.choice(list(VENDORS.keys()))
        vendor_info = VENDORS[vendor_name]
        category = vendor_info["category"]
        platform = random.choices(PLATFORMS, weights=[0.55, 0.30, 0.15])[0]
        payment_method = random.choice(PAYMENT_METHODS)

        # Time distribution: 85% business hours (9-20), 15% off-hours
        day_offset = random.randint(0, 75)
        if random.random() < 0.85:
            hour = random.randint(9, 19)
        else:
            hour = random.choice([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        timestamp = base_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        # Baseline amount from normal distribution
        raw_amount = max(100.0, np.random.normal(vendor_info["mean"], vendor_info["std"]))
        amount = round(raw_amount, 2)

        is_anomaly = i in anomaly_indices
        anomaly_type = "NORMAL"

        if is_anomaly:
            scenario = random.choice(["typo_extra_zero", "off_hours_high_spend", "vendor_amount_spike", "duplicate_invoice"])
            if scenario == "typo_extra_zero":
                # User typed 18000 instead of 1800
                amount = round(amount * 10, 2)
                anomaly_type = "TYPO_EXTRA_ZERO"
            elif scenario == "off_hours_high_spend":
                # High spend at 2:00 AM on Sunday
                timestamp = timestamp.replace(hour=random.choice([1, 2, 3, 4]))
                amount = round(amount * 3.5, 2)
                anomaly_type = "OFF_HOURS_HIGH_SPEND"
            elif scenario == "vendor_amount_spike":
                # Sudden 4-5x increase over vendor baseline
                amount = round(amount * 4.2, 2)
                anomaly_type = "VENDOR_AMOUNT_SPIKE"
            elif scenario == "duplicate_invoice":
                # Will duplicate a prior transaction if available
                if len(records) > 5:
                    prev_tx = random.choice(records[-10:])
                    vendor_name = prev_tx["vendor"]
                    category = prev_tx["category"]
                    amount = prev_tx["amount"]
                    timestamp = prev_tx["timestamp"] + timedelta(hours=random.randint(2, 36))
                anomaly_type = "DUPLICATE_INVOICE"

        records.append({
            "transaction_id": tx_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "vendor": vendor_name,
            "category": category,
            "amount": amount,
            "payment_method": payment_method,
            "platform": platform,
            "is_anomaly_ground_truth": int(is_anomaly),
            "anomaly_type": anomaly_type
        })

    df = pd.DataFrame(records)
    # Sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "transactions.csv")
    print(f"Generating synthetic SMB bookkeeping dataset...")
    df = generate_transactions(num_records=5000, anomaly_rate=0.05)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} transactions to {output_path}")
    print(f"Anomalies injected: {df['is_anomaly_ground_truth'].sum()} ({df['is_anomaly_ground_truth'].mean():.1%})")
    print("Anomaly breakdown:")
    print(df[df['is_anomaly_ground_truth'] == 1]['anomaly_type'].value_counts())
