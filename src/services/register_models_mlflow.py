"""Registers all 4 existing trained models into MLflow with metadata, metrics, and 'champion' alias."""
import json
import logging
import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("register_mlflow")

MODELS_CONFIG = [
    {
        "experiment_name": "Demand Forecasting",
        "registered_model_name": "DemandForecastingModel",
        "artifact_path": "models/demand_forecasting/demand_model.joblib",
        "meta_dir": "chainsight_ai_complete_models/demand_forecasting",
        "flavor": "lightgbm",
        "task": "Time-Series Tabular Regression",
    },
    {
        "experiment_name": "Stockout Prediction",
        "registered_model_name": "StockoutPredictionModel",
        "artifact_path": "models/stockout_prediction/stockout_model.joblib",
        "meta_dir": "chainsight_ai_complete_models/stockout_prediction",
        "flavor": "sklearn",
        "task": "Binary Classification",
    },
    {
        "experiment_name": "Delivery ETA Prediction",
        "registered_model_name": "DeliveryETAModel",
        "artifact_path": "models/delivery_eta/delivery_eta_model.joblib",
        "meta_dir": "chainsight_ai_complete_models/delivery_eta",
        "flavor": "catboost",
        "task": "Tabular Regression",
    },
    {
        "experiment_name": "Seller Risk Prediction",
        "registered_model_name": "SellerRiskModel",
        "artifact_path": "models/seller_risk/seller_risk_model.joblib",
        "meta_dir": "chainsight_ai_complete_models/seller_risk",
        "flavor": "catboost",
        "task": "Binary Classification",
    },
]

def load_json(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def register_all():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("Using MLflow Tracking URI: %s", tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    for cfg in MODELS_CONFIG:
        exp_name = cfg["experiment_name"]
        reg_name = cfg["registered_model_name"]
        logger.info("Processing: %s -> %s", exp_name, reg_name)

        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            exp_id = client.create_experiment(exp_name)
        else:
            exp_id = exp.experiment_id

        # Load metadata and metrics
        metrics = load_json(os.path.join(cfg["meta_dir"], "metrics.json"))
        training_meta = load_json(os.path.join(cfg["meta_dir"], "training_metadata.json"))
        feature_cols = load_json(os.path.join(cfg["meta_dir"], "feature_columns.json"))

        model_obj = joblib.load(cfg["artifact_path"])

        with mlflow.start_run(experiment_id=exp_id, run_name=f"import_{reg_name}_v1") as run:
            # 1. Log tags
            mlflow.set_tag("task", cfg["task"])
            mlflow.set_tag("provenance", "imported_historical_model")
            mlflow.set_tag("original_version", training_meta.get("version", "1.0.0"))
            if "training_date_range" in training_meta:
                mlflow.set_tag("training_date_range", training_meta["training_date_range"])

            # 2. Log parameters
            params = {
                "algorithm": training_meta.get("estimator_name", cfg["flavor"]),
                "feature_count": len(feature_cols) if isinstance(feature_cols, list) else len(feature_cols.get("feature_names", [])),
            }
            if "hyperparameters" in training_meta:
                hp = training_meta["hyperparameters"]
                if isinstance(hp, dict):
                    for k, v in list(hp.items())[:15]:
                        if v is not None:
                            params[k] = str(v)
            mlflow.log_params(params)

            # 3. Log metrics
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    clean_metrics[k] = float(v)
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)

            # 4. Log model artifact with appropriate flavor and cloudpickle serialization
            if cfg["flavor"] == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model_obj,
                    name="model",
                    registered_model_name=reg_name,
                    serialization_format="cloudpickle"
                )
            elif cfg["flavor"] == "lightgbm":
                mlflow.lightgbm.log_model(
                    lgb_model=model_obj,
                    name="model",
                    registered_model_name=reg_name
                )
            elif cfg["flavor"] == "catboost":
                mlflow.catboost.log_model(
                    cb_model=model_obj,
                    name="model",
                    registered_model_name=reg_name
                )

            logger.info("Logged run %s for %s", run.info.run_id, reg_name)

        # 5. Set 'champion' alias on latest version
        try:
            versions = client.search_model_versions(f"name='{reg_name}'")
            if versions:
                latest_version = max(int(v.version) for v in versions)
                client.set_registered_model_alias(
                    name=reg_name,
                    alias="champion",
                    version=str(latest_version)
                )
                logger.info("Successfully assigned alias 'champion' -> %s version %s", reg_name, latest_version)
        except Exception as e:
            logger.warning("Could not set alias for %s: %s", reg_name, e)

    logger.info("MLflow model registration complete for all 4 models.")

if __name__ == "__main__":
    register_all()
