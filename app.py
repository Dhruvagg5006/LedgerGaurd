"""
app.py - Simple FastAPI microservice for real-time SMB expense verification.
Accepts an expense and returns whether it should be flagged for review,
with an anomaly probability and a human-readable explanation.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="LedgerGuard API",
    description="Simple AI/ML microservice to detect anomalous SMB expenses",
    version="1.0.0"
)

@app.get("/", include_in_schema=False)
def root():
    """Redirect home requests directly to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")


# Load saved model bundle on startup
MODEL_PATH = "model.joblib"
bundle = None

if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    print("Loaded model.joblib successfully!")
else:
    print("Warning: model.joblib not found. Run 'python train.py' first.")


class ExpenseRequest(BaseModel):
    category: str = Field(..., example="Office Supplies")
    amount: float = Field(..., gt=0, example=12500.0)
    hour: int = Field(..., ge=0, le=23, example=22)
    is_weekend: int = Field(0, ge=0, le=1, example=1)


class ExpenseResponse(BaseModel):
    flag_for_review: bool
    risk_level: str
    anomaly_probability: float
    category_baseline_avg: float
    amount_ratio: float
    explanation: str


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": bundle is not None}


@app.post("/predict", response_model=ExpenseResponse)
def predict_expense(expense: ExpenseRequest):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train first.")

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    category_means = bundle["category_means"]

    # Calculate engineered features
    cat_avg = category_means.get(expense.category, expense.amount)
    amount_ratio = round(expense.amount / (cat_avg + 1e-5), 2)
    is_off_hours = 1 if (expense.hour < 8 or expense.hour >= 20) else 0

    # Build input row matching training features
    input_data = {col: 0 for col in feature_cols}
    input_data["amount"] = expense.amount
    input_data["hour"] = expense.hour
    input_data["is_weekend"] = expense.is_weekend
    input_data["amount_to_cat_ratio"] = amount_ratio
    input_data["is_off_hours"] = is_off_hours

    # Set one-hot category column if present
    cat_col = f"category_{expense.category}"
    if cat_col in input_data:
        input_data[cat_col] = 1

    input_df = pd.DataFrame([input_data])[feature_cols]

    # Model inference
    proba = float(model.predict_proba(input_df)[0, 1])
    is_flagged = bool(proba >= 0.50)

    # Risk level categorization
    if proba >= 0.75:
        risk_level = "HIGH"
    elif proba >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Human-readable explanation
    reasons = []
    if amount_ratio >= 3.0:
        reasons.append(f"Amount is {amount_ratio}x higher than typical {expense.category} expense (₹{cat_avg:.0f})")
    if is_off_hours:
        reasons.append(f"Logged at off-hour ({expense.hour:02d}:00)")
    if expense.is_weekend:
        reasons.append("Logged during weekend")

    if not reasons:
        explanation = "Transaction fits typical spending patterns for this category."
    else:
        explanation = " | ".join(reasons)

    return ExpenseResponse(
        flag_for_review=is_flagged,
        risk_level=risk_level,
        anomaly_probability=round(proba, 4),
        category_baseline_avg=round(cat_avg, 2),
        amount_ratio=amount_ratio,
        explanation=explanation
    )
