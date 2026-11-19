"""Seller Delay Risk Prediction Service (CatBoost Classifier)."""
from datetime import datetime, timezone
import time
from typing import Any, Dict
import pandas as pd
from src.services.base import BaseModelService

FEATURE_COLUMNS = [
    "price", "freight_value", "product_weight_g",
    "product_category_name", "seller_state", "customer_state",
    "seller_delivery_count"
]

class SellerRiskService(BaseModelService):
    def __init__(self, local_path: str = "models/seller_risk/seller_risk_model.joblib"):
        super().__init__(
            model_name="SellerRiskModel",
            local_path=local_path,
            mlflow_model_name="SellerRiskModel",
        )

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        df = pd.DataFrame([{col: payload[col] for col in FEATURE_COLUMNS}])

        probabilities = self.model.predict_proba(df)[0]
        prob_on_time = float(probabilities[0])
        prob_delayed = float(probabilities[1])

        # Prediction: 1 = High risk of delay, 0 = Low risk
        predicted_class = int(self.model.predict(df)[0])
        label = "High Delay Risk" if predicted_class == 1 else "Normal Risk"

        if prob_delayed >= 0.60:
            risk_level = "HIGH"
        elif prob_delayed >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": self.model_name,
            "version": self.version,
            "source": self.source,
            "prediction": predicted_class,
            "prediction_label": label,
            "delay_probability": round(prob_delayed, 4),
            "on_time_probability": round(prob_on_time, 4),
            "risk_level": risk_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "metadata": {
                "algorithm": "CatBoostClassifier",
                "note": "Class balanced CatBoost model trained on Olist delivery history"
            }
        }
