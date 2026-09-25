"""
demo.py - Quick script to test the running FastAPI server.
Run this in a separate terminal while uvicorn is running:
python demo.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

print(f"Connecting to {BASE_URL}...\n")

# 1. Test Health
health = requests.get(f"{BASE_URL}/health").json()
print("Health Status:", health)

# 2. Test Normal Transaction
normal_tx = {
    "category": "Office Supplies",
    "amount": 1200.0,
    "hour": 14,
    "is_weekend": 0
}
print("\n--- Test 1: Normal Transaction ---")
res1 = requests.post(f"{BASE_URL}/predict", json=normal_tx).json()
print("Input:", normal_tx)
print("Output:", res1)

# 3. Test Anomalous Transaction (Typo: 15,000 for office supplies at 11 PM)
anomaly_tx = {
    "category": "Office Supplies",
    "amount": 15000.0,
    "hour": 23,
    "is_weekend": 1
}
print("\n--- Test 2: Anomalous Transaction ---")
res2 = requests.post(f"{BASE_URL}/predict", json=anomaly_tx).json()
print("Input:", anomaly_tx)
print("Output:", res2)
