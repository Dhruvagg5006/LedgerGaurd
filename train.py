"""
train.py - Trains a simple, lightweight ML model to flag unusual SMB expenses.
Demonstrates:
1. Data Preprocessing & Feature Engineering (Pandas)
2. Handling Class Imbalance (Stratified Split & Class Weight)
3. Model Training (Random Forest)
4. Evaluation Metrics (Precision, Recall, F1, ROC-AUC)
5. Model Persistence (joblib)
"""

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_model():
    print("Step 1: Loading transactions data...")
    df = pd.read_csv("transactions.csv")
    
    # Calculate baseline average per category
    category_means = df[df["requires_review"] == 0].groupby("category")["amount"].mean().to_dict()
    with open("category_means.json", "w") as f:
        json.dump(category_means, f, indent=2)
    print(f"Saved category baseline stats for {len(category_means)} categories.")

    # Step 2: Feature Engineering
    # Simple, explainable feature: ratio of transaction amount to typical category average
    df["amount_to_cat_ratio"] = df.apply(
        lambda row: row["amount"] / (category_means.get(row["category"], row["amount"]) + 1e-5),
        axis=1
    )
    # Binary flag for off-hours (before 8 AM or after 8 PM)
    df["is_off_hours"] = df["hour"].apply(lambda h: 1 if (h < 8 or h >= 20) else 0)

    # Encode categories using simple one-hot encoding
    df_encoded = pd.get_dummies(df, columns=["category"], drop_first=True)

    # Feature columns and target
    feature_cols = [col for col in df_encoded.columns if col not in ["requires_review"]]
    X = df_encoded[feature_cols]
    y = df_encoded["requires_review"]

    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")

    # Step 3: Train / Test Split (Stratified to maintain 10% anomaly proportion)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Step 4: Model Training
    # RandomForest with class_weight='balanced' so the model focuses on the rare anomaly class
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Step 5: Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- MODEL EVALUATION RESULTS ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Requires Review"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\nTN={cm[0,0]}  FP={cm[0,1]}\nFN={cm[1,0]}  TP={cm[1,1]}")

    # Feature Importances
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop Feature Importances:")
    print(importances.head(4).to_string())

    # Step 6: Save Model and Metadata
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "category_means": category_means
    }
    joblib.dump(bundle, "model.joblib")
    print("\nSaved trained model to 'model.joblib'. Ready for deployment!")

if __name__ == "__main__":
    train_model()
