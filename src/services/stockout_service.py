"""Stockout Prediction inference service (Scikit-Learn Pipeline with LightGBM Classifier)."""
from datetime import datetime, timezone
import time
from typing import Any, Dict
import pandas as pd
from src.services.base import BaseModelService

FEATURE_COLUMNS = [
    "opening_stock", "received_stock", "avg_daily_demand_7d",
    "avg_daily_demand_30d", "demand_std_7d", "inventory_coverage_days",
    "supplier_avg_lead_time", "supplier_reliability", "warehouse_utilization_pct",
    "warehouse_id", "product_category", "day_of_week", "month"
]

OPTIMAL_F1_THRESHOLD = 0.45560

class StockoutPredictionService(BaseModelService):
    def __init__(self, local_path: str = "models/stockout_prediction/stockout_model.joblib"):
        super().__init__(
            model_name="StockoutPredictionModel",
            local_path=local_path,
            mlflow_model_name="StockoutPredictionModel",
        )

    def preprocess(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Constructs and validates DataFrame for the self-contained pipeline."""
        row = {col: payload[col] for col in FEATURE_COLUMNS}
        df = pd.DataFrame([row])

        # Validate types
        for int_col in ["opening_stock", "received_stock", "supplier_avg_lead_time", "day_of_week", "month"]:
            df[int_col] = df[int_col].astype(int)

        for float_col in [
            "avg_daily_demand_7d", "avg_daily_demand_30d", "demand_std_7d",
            "inventory_coverage_days", "supplier_reliability", "warehouse_utilization_pct"
        ]:
            df[float_col] = df[float_col].astype(float)

        df["warehouse_id"] = df["warehouse_id"].astype(str)
        df["product_category"] = df["product_category"].astype(str)

        return df[FEATURE_COLUMNS]

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        df = self.preprocess(payload)

        # Scikit-learn Pipeline handles scaling and one-hot encoding end-to-end
        probabilities = self.model.predict_proba(df)[0]
        prob_no_stockout = float(probabilities[0])
        prob_stockout = float(probabilities[1])

        # Apply calibrated F1 threshold from training metrics
        predicted_class = 1 if prob_stockout >= OPTIMAL_F1_THRESHOLD else 0
        status_label = "Stockout Risk Detected" if predicted_class == 1 else "Normal Inventory Level"

        # Risk severity classification
        if prob_stockout >= 0.75:
            severity = "CRITICAL"
        elif prob_stockout >= OPTIMAL_F1_THRESHOLD:
            severity = "HIGH"
        elif prob_stockout >= 0.25:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": self.model_name,
            "version": self.version,
            "source": self.source,
            "prediction": predicted_class,
            "prediction_label": status_label,
            "stockout_probability": round(prob_stockout, 4),
            "no_stockout_probability": round(prob_no_stockout, 4),
            "risk_severity": severity,
            "calibrated_threshold": OPTIMAL_F1_THRESHOLD,
            "horizon": "next_7_days",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "metadata": {
                "algorithm": "Pipeline(ColumnTransformer + LGBMClassifier)",
                "classes": ["No Stockout", "Stockout"],
                "training_f1_score": 0.9405
            }
        }
