"""
LedgerGuard Feature Engineering Pipeline
Extracts statistical, temporal, and velocity signals for SMB bookkeeping anomaly detection.
Handles cold-start vendors by falling back to category-level baselines.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class TransactionFeatureExtractor:
    """
    Computes statistical baselines per vendor and category, and transforms
    raw transactions into ML-ready numerical feature vectors.
    """

    def __init__(self):
        self.vendor_stats: Dict[str, Dict[str, float]] = {}
        self.category_stats: Dict[str, Dict[str, float]] = {}
        self.global_stats: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "TransactionFeatureExtractor":
        """
        Fit historical baseline statistics from training transaction data.
        """
        df = df.copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

        # 1. Global baseline stats
        self.global_stats = {
            "mean": float(df["amount"].mean()),
            "std": float(df["amount"].std() or 1.0),
            "median": float(df["amount"].median()),
        }

        # 2. Category-level stats (fallback for cold-start vendors)
        for cat, group in df.groupby("category"):
            amounts = group["amount"]
            std = float(amounts.std())
            self.category_stats[cat] = {
                "mean": float(amounts.mean()),
                "std": std if std > 0 else 1.0,
                "median": float(amounts.median()),
                "count": len(amounts),
            }

        # 3. Vendor-level stats
        for vendor, group in df.groupby("vendor"):
            amounts = group["amount"]
            std = float(amounts.std())
            self.vendor_stats[vendor] = {
                "mean": float(amounts.mean()),
                "std": std if std > 0 else 1.0,
                "median": float(amounts.median()),
                "count": len(amounts),
                "category": group["category"].iloc[0],
            }

        self.is_fitted = True
        return self

    def extract_single(
        self,
        tx: Dict[str, Any],
        recent_user_txs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract features for a single transaction (used by real-time API).
        """
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transforming.")

        amount = float(tx.get("amount", 0.0))
        vendor = str(tx.get("vendor", "")).strip()
        category = str(tx.get("category", "")).strip()
        timestamp = pd.to_datetime(tx.get("timestamp", pd.Timestamp.now()))

        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        is_off_hours = 1 if (hour < 8 or hour >= 21) else 0
        is_weekend = 1 if day_of_week >= 5 else 0

        # Baseline resolution (Vendor -> Category Fallback -> Global Fallback)
        is_cold_start_vendor = False
        if vendor in self.vendor_stats and self.vendor_stats[vendor]["count"] >= 3:
            stats = self.vendor_stats[vendor]
        elif category in self.category_stats:
            stats = self.category_stats[category]
            is_cold_start_vendor = True
        else:
            stats = self.global_stats
            is_cold_start_vendor = True

        v_mean = stats["mean"]
        v_std = stats["std"]
        v_median = stats["median"]

        amount_zscore = (amount - v_mean) / (v_std + 1e-5)
        amount_ratio_median = amount / (v_median + 1e-5)
        log_amount = float(np.log1p(max(0.0, amount)))

        # Velocity & duplicate checks against recent transactions
        is_duplicate = 0
        velocity_24h = 0
        hours_since_last = 999.0

        if recent_user_txs:
            for past in recent_user_txs:
                past_time = pd.to_datetime(past.get("timestamp"))
                diff_hours = (timestamp - past_time).total_seconds() / 3600.0

                if 0 <= diff_hours <= 24:
                    if past.get("vendor") == vendor:
                        velocity_24h += 1

                if 0 < diff_hours <= 48:
                    if past.get("vendor") == vendor:
                        hours_since_last = min(hours_since_last, diff_hours)
                        # Check duplicate: within 2% amount
                        past_amount = float(past.get("amount", 0.0))
                        if abs(amount - past_amount) <= max(1.0, 0.02 * past_amount):
                            is_duplicate = 1

        # Platform encoding
        platform = str(tx.get("platform", "android")).lower()
        is_platform_android = 1 if platform == "android" else 0
        is_platform_ios = 1 if platform == "ios" else 0
        is_platform_web = 1 if platform == "web" else 0

        features = {
            "amount": amount,
            "log_amount": log_amount,
            "amount_zscore": round(float(amount_zscore), 4),
            "amount_ratio_median": round(float(amount_ratio_median), 4),
            "hour": hour,
            "is_off_hours": is_off_hours,
            "is_weekend": is_weekend,
            "is_cold_start_vendor": 1 if is_cold_start_vendor else 0,
            "velocity_24h": velocity_24h,
            "is_duplicate_candidate": is_duplicate,
            "is_platform_android": is_platform_android,
            "is_platform_ios": is_platform_ios,
            "is_platform_web": is_platform_web,
        }
        return features

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch transformation on historical dataframe.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        feature_rows = []
        user_history: Dict[str, List[Dict[str, Any]]] = {}

        for _, row in df.iterrows():
            tx_dict = row.to_dict()
            u_id = row["user_id"]
            recent = user_history.get(u_id, [])

            feats = self.extract_single(tx_dict, recent_user_txs=recent)
            feature_rows.append(feats)

            # Maintain sliding history of last 10 transactions per user
            if u_id not in user_history:
                user_history[u_id] = []
            user_history[u_id].append({"timestamp": row["timestamp"], "vendor": row["vendor"], "amount": row["amount"]})
            if len(user_history[u_id]) > 15:
                user_history[u_id].pop(0)

        feature_df = pd.DataFrame(feature_rows)
        return feature_df


FEATURE_COLUMNS = [
    "log_amount",
    "amount_zscore",
    "amount_ratio_median",
    "hour",
    "is_off_hours",
    "is_weekend",
    "is_cold_start_vendor",
    "velocity_24h",
    "is_duplicate_candidate",
    "is_platform_android",
    "is_platform_ios",
    "is_platform_web",
]
