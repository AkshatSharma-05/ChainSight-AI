"""Monitoring and Data Drift Detection Engine for ChainSight-AI."""
from collections import deque
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional
import numpy as np

# Baseline reference statistics recovered from training metadata (e.g. Stockout model scaler)
STOCKOUT_REFERENCE_STATS = {
    "opening_stock": {"mean": 886.86, "std": 1396.21},
    "received_stock": {"mean": 20.55, "std": 119.48},
    "avg_daily_demand_7d": {"mean": 19.73, "std": 16.12},
    "warehouse_utilization_pct": {"mean": 569.75, "std": 257.18},
    "supplier_reliability": {"mean": 0.81, "std": 0.07}
}

class SimpleDriftMonitor:
    """Tracks live inference metrics and monitors input feature distributions for statistical drift."""

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.inference_log: deque = deque(maxlen=window_size)
        self.predictions_by_model: Dict[str, deque] = {
            "DemandForecastingModel": deque(maxlen=window_size),
            "StockoutPredictionModel": deque(maxlen=window_size),
            "DeliveryETAModel": deque(maxlen=window_size),
            "SellerRiskModel": deque(maxlen=window_size),
        }
        self.feature_streams: Dict[str, deque] = {
            feature: deque(maxlen=window_size) for feature in STOCKOUT_REFERENCE_STATS
        }

    def record_inference(
        self,
        model_name: str,
        prediction: Any,
        latency_ms: float,
        features: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records an inference event for performance tracking and drift monitoring."""
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "model_name": model_name,
            "prediction": prediction,
            "latency_ms": latency_ms,
        }
        self.inference_log.append(record)

        if model_name in self.predictions_by_model:
            self.predictions_by_model[model_name].append(prediction)

        if features:
            for feat_name, val in features.items():
                if feat_name in self.feature_streams and isinstance(val, (int, float)):
                    self.feature_streams[feat_name].append(float(val))

    def detect_drift(self) -> Dict[str, Any]:
        """Calculates drift metrics comparing live sliding window against training baseline."""
        drift_report = {}
        drift_detected_count = 0

        for feature, baseline in STOCKOUT_REFERENCE_STATS.items():
            stream = list(self.feature_streams[feature])
            sample_count = len(stream)

            if sample_count < 10:
                drift_report[feature] = {
                    "status": "INSUFFICIENT_DATA",
                    "sample_count": sample_count,
                    "min_required": 10,
                }
                continue

            current_mean = float(np.mean(stream))
            current_std = float(np.std(stream)) if np.std(stream) > 1e-6 else 1e-6

            # Standardized Z-Score difference of means
            # z = (current_mean - baseline_mean) / (baseline_std / sqrt(N))
            standard_error = baseline["std"] / math.sqrt(sample_count)
            z_score = abs(current_mean - baseline["mean"]) / standard_error

            # Simple drift decision rule (Z > 3.0 represents p < 0.003 statistical drift)
            if z_score > 3.0:
                status = "DRIFT_DETECTED"
                drift_detected_count += 1
            elif z_score > 2.0:
                status = "DRIFT_WARNING"
            else:
                status = "NO_DRIFT"

            drift_report[feature] = {
                "status": status,
                "z_score": round(z_score, 2),
                "baseline_mean": baseline["mean"],
                "current_mean": round(current_mean, 2),
                "sample_count": sample_count
            }

        overall_status = "DRIFT_DETECTED" if drift_detected_count > 0 else "HEALTHY"

        return {
            "monitoring_status": overall_status,
            "total_requests_tracked": len(self.inference_log),
            "window_size": self.window_size,
            "feature_drift_analysis": drift_report,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Returns latency and throughput metrics across all inference endpoints."""
        total = len(self.inference_log)
        if total == 0:
            return {"total_inferences": 0, "avg_latency_ms": 0.0}

        latencies = [entry["latency_ms"] for entry in self.inference_log]
        return {
            "total_inferences": total,
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "max_latency_ms": round(float(np.max(latencies)), 2),
            "min_latency_ms": round(float(np.min(latencies)), 2),
        }

# Global singleton monitor
monitor = SimpleDriftMonitor()
