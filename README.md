# LedgerGuard — SMB Expense Anomaly Detector

A lightweight, end-to-end Machine Learning microservice built to flag unusual and accidental expense entries in SMB bookkeeping and mobile accounting apps.

---

## 1. Problem Statement
Small business owners and bookkeepers often record expenses quickly from mobile devices. Common human errors include:
- **Typo / Order-of-magnitude errors:** Logging ₹15,000 instead of ₹1,500 for routine stationery.
- **Unusual off-hours charges:** High expenditures recorded late at night or over weekends.
- **Out-of-pattern spending:** Sudden multi-fold spikes compared to historical category baselines.

Rather than relying on complex black-box neural networks, **LedgerGuard** uses an interpretable **Random Forest classifier** paired with statistical category baselines and exposes a real-time **FastAPI** microservice.

---

## 2. Project Structure

```text
ledger-guard/
│
├── data.py              # Generates synthetic SMB transaction dataset (1,200 records)
├── train.py             # Feature engineering, model training, evaluation & serialization
├── app.py               # FastAPI microservice with /predict and /health
├── test_app.py          # Automated verification tests for API and model
├── transactions.csv     # Training data
├── category_means.json  # Historical category baselines
├── model.joblib         # Saved model artifact
├── requirements.txt     # Dependencies
└── README.md            # Technical documentation
```

---

## 3. How to Run (Step-by-Step)

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Generate Data & Train Model
```bash
python data.py
python train.py
```
*Outputs evaluation metrics: Precision, Recall, F1-score, ROC-AUC, and saves `model.joblib`.*

### Step 3: Run the Automated Tests
```bash
python test_app.py
```

### Step 4: Launch the FastAPI Server
```bash
uvicorn app:app --reload --port 8000
```
Visit the interactive Swagger UI documentation at: `http://localhost:8000/docs`

---

## 4. API Usage Example

### Request: `POST /predict`
```json
{
  "category": "Office Supplies",
  "amount": 15000.0,
  "hour": 23,
  "is_weekend": 1
}
```

### Response:
```json
{
  "flag_for_review": true,
  "risk_level": "HIGH",
  "anomaly_probability": 0.89,
  "category_baseline_avg": 1498.2,
  "amount_ratio": 10.01,
  "explanation": "Amount is 10.01x higher than typical Office Supplies expense (₹1498) | Logged at off-hour (23:00) | Logged during weekend"
}
```

---

## 5. Key Engineering Decisions (Interview Talking Points)

1. **Why Random Forest over Deep Learning?**
   - Tabular business transaction data is best modeled by decision trees.
   - Low latency ($< 10\text{ ms}$ inference), zero GPU dependency, and high explainability.

2. **Handling Class Imbalance:**
   - Real-world anomalies represent only $\sim 10\%$ of transactions.
   - We used **stratified sampling** during train/test split and set `class_weight='balanced'` in the model so it prioritizes detecting rare anomalies without sacrificing precision.

3. **Domain-Specific Feature Engineering:**
   - Rather than feeding raw amounts, the key signal is `amount_to_cat_ratio` ($\frac{\text{amount}}{\text{historical category average}}$), which accounts for ~60% of model feature importance.

4. **Actionable Explainability:**
   - A product team cannot act on just a probability score. The API provides human-readable `explanation` strings explaining *why* an expense was flagged, ready to display directly in the mobile UI.
