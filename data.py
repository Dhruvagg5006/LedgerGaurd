"""
data.py - Generates a clean, simple dataset of SMB transactions.
Features: category, amount, hour, is_weekend, requires_review (target: 0 or 1)
"""

import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

CATEGORIES = {
    "Office Supplies": {"mean": 1500, "std": 400},
    "Travel & Commute": {"mean": 600, "std": 200},
    "Hosting & Software": {"mean": 8000, "std": 1500},
    "Food & Meals": {"mean": 750, "std": 250},
    "Utilities": {"mean": 3500, "std": 800},
    "Contractors": {"mean": 25000, "std": 5000},
}

def generate_dataset(n_samples: int = 1200) -> pd.DataFrame:
    records = []
    
    for _ in range(n_samples):
        cat = random.choice(list(CATEGORIES.keys()))
        cat_info = CATEGORIES[cat]
        
        # 90% normal transactions, 10% anomalies
        is_anomaly = 1 if random.random() < 0.10 else 0
        
        if is_anomaly:
            # Anomalies: typo (10x amount), or weird off-hours / weekend spikes
            anomaly_type = random.choice(["typo", "unusual_amount", "off_hours"])
            if anomaly_type == "typo":
                amount = round(cat_info["mean"] * random.uniform(5.0, 10.0), 2)
                hour = random.randint(9, 18)
                is_weekend = 0
            elif anomaly_type == "unusual_amount":
                amount = round(cat_info["mean"] * random.uniform(3.5, 6.0), 2)
                hour = random.randint(0, 23)
                is_weekend = random.choice([0, 1])
            else:  # off-hours
                amount = round(cat_info["mean"] * random.uniform(1.5, 3.0), 2)
                hour = random.choice([1, 2, 3, 4, 23])
                is_weekend = 1
        else:
            # Normal transactions
            raw_amount = max(100.0, np.random.normal(cat_info["mean"], cat_info["std"]))
            amount = round(raw_amount, 2)
            hour = random.choice([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
            is_weekend = 1 if random.random() < 0.1 else 0

        records.append({
            "category": cat,
            "amount": amount,
            "hour": hour,
            "is_weekend": is_weekend,
            "requires_review": is_anomaly
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = generate_dataset(1200)
    df.to_csv("transactions.csv", index=False)
    print("Generated transactions.csv with 1200 rows.")
    print("Class distribution:")
    print(df["requires_review"].value_counts(normalize=True))
