"""
test_app.py - Simple unit tests for the ML model and FastAPI endpoints.
Can be run with: pytest test_app.py
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict_normal_expense():
    # Normal office supplies: ₹1,200 during work hours
    payload = {
        "category": "Office Supplies",
        "amount": 1200.0,
        "hour": 14,
        "is_weekend": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["flag_for_review"] is False
    assert data["risk_level"] == "LOW"
    assert data["anomaly_probability"] < 0.5

def test_predict_anomalous_expense():
    # Typo / massive expense: ₹15,000 for office supplies (normally ~₹1,500) at 11 PM
    payload = {
        "category": "Office Supplies",
        "amount": 15000.0,
        "hour": 23,
        "is_weekend": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["flag_for_review"] is True
    assert data["risk_level"] in ["MEDIUM", "HIGH"]
    assert data["anomaly_probability"] >= 0.5
    assert "higher than typical" in data["explanation"]

if __name__ == "__main__":
    print("Running tests...")
    test_health_endpoint()
    print("[PASS] test_health_endpoint passed")
    test_predict_normal_expense()
    print("[PASS] test_predict_normal_expense passed")
    test_predict_anomalous_expense()
    print("[PASS] test_predict_anomalous_expense passed")
    print("\nAll 3 tests passed successfully!")

