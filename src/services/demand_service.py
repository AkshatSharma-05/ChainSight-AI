"""Demand Forecasting inference service (Walmart M5 LightGBM Regressor)."""
from datetime import datetime, timezone
import time
from typing import Any, Dict
import pandas as pd
from src.services.base import BaseModelService

DEPT_CATEGORIES = [
    "FOODS_1", "FOODS_2", "FOODS_3",
    "HOBBIES_1", "HOBBIES_2",
    "HOUSEHOLD_1", "HOUSEHOLD_2"
]

CAT_CATEGORIES = [
    "FOODS", "HOBBIES", "HOUSEHOLD"
]

FEATURE_COLUMNS = [
    "wday", "month", "year", "sell_price",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
    "dept_id", "cat_id"
]

class DemandForecastingService(BaseModelService):
    def __init__(self, local_path: str = "models/demand_forecasting/demand_model.joblib"):
        super().__init__(
            model_name="DemandForecastingModel",
            local_path=local_path,
            mlflow_model_name="DemandForecastingModel",
        )

    def preprocess(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Constructs and casts DataFrame to match LightGBM booster's categorical schema."""
        row = {col: payload[col] for col in FEATURE_COLUMNS}
        df = pd.DataFrame([row])

        # Validate and apply exact pandas Categorical dtypes required by LightGBM
        dept = str(payload["dept_id"]).upper()
        cat = str(payload["cat_id"]).upper()

        if dept not in DEPT_CATEGORIES:
            raise ValueError(
                f"Invalid dept_id '{payload['dept_id']}'. Allowed values: {DEPT_CATEGORIES}"
            )
        if cat not in CAT_CATEGORIES:
            raise ValueError(
                f"Invalid cat_id '{payload['cat_id']}'. Allowed values: {CAT_CATEGORIES}"
            )

        df["dept_id"] = pd.Categorical([dept], categories=DEPT_CATEGORIES)
        df["cat_id"] = pd.Categorical([cat], categories=CAT_CATEGORIES)

        # Ensure correct numerical dtypes
        df["wday"] = df["wday"].astype(int)
        df["month"] = df["month"].astype(int)
        df["year"] = df["year"].astype(int)
        for num_col in ["sell_price", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28", "rolling_std_7"]:
            df[num_col] = df[num_col].astype(float)

        return df[FEATURE_COLUMNS]

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        df = self.preprocess(payload)

        raw_pred = self.model.predict(df)
        predicted_val = float(raw_pred[0])
        # Demand cannot be negative in physical inventory
        bounded_demand = max(0.0, round(predicted_val, 2))
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": self.model_name,
            "version": self.version,
            "source": self.source,
            "prediction": bounded_demand,
            "unit": "units",
            "horizon": "next_7_days",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "metadata": {
                "algorithm": "LightGBMRegressor",
                "features_used": len(FEATURE_COLUMNS),
                "target_definition": "Target unit sales for next 7 days"
            }
        }
