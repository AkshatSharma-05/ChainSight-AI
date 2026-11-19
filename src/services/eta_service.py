"""Delivery ETA Prediction Service (CatBoost Regressor)."""
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

class DeliveryETAService(BaseModelService):
    def __init__(self, local_path: str = "models/delivery_eta/delivery_eta_model.joblib"):
        super().__init__(
            model_name="DeliveryETAModel",
            local_path=local_path,
            mlflow_model_name="DeliveryETAModel",
        )

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        df = pd.DataFrame([{col: payload[col] for col in FEATURE_COLUMNS}])

        # CatBoost natively handles categorical features
        raw_pred = self.model.predict(df)
        eta_days = max(1.0, round(float(raw_pred[0]), 1))
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": self.model_name,
            "version": self.version,
            "source": self.source,
            "prediction": eta_days,
            "unit": "days",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "metadata": {
                "algorithm": "CatBoostRegressor",
                "training_mae": 6.16
            }
        }
