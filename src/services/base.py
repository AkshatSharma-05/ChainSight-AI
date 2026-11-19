"""Base model service providing unified interface and lifecycle management."""
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import joblib

logger = logging.getLogger("chainsight.services")

class BaseModelService(ABC):
    """Abstract base class for all machine learning inference services."""

    def __init__(self, model_name: str, local_path: str, mlflow_model_name: Optional[str] = None):
        self.model_name = model_name
        self.local_path = local_path
        self.mlflow_model_name = mlflow_model_name
        self.model: Any = None
        self.version: str = "1.0.0"
        self.source: str = "local"
        self.load_model()

    def load_model(self) -> None:
        """Loads model from MLflow registry if configured and available, else from local joblib."""
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        use_mlflow = os.getenv("USE_MLFLOW_REGISTRY", "false").lower() in ("true", "1")

        if use_mlflow and mlflow_uri and self.mlflow_model_name:
            try:
                import mlflow
                model_uri = f"models:/{self.mlflow_model_name}@champion"
                logger.info("Attempting to load %s from MLflow: %s", self.model_name, model_uri)
                self.model = mlflow.pyfunc.load_model(model_uri)
                self.source = "mlflow_registry"
                logger.info("Successfully loaded %s from MLflow registry.", self.model_name)
                return
            except Exception as exc:
                logger.warning(
                    "Could not load %s from MLflow registry (%s). Falling back to local artifact: %s",
                    self.model_name, exc, self.local_path
                )

        # Local joblib loading
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(
                f"Model artifact for '{self.model_name}' not found at {self.local_path}."
            )
        self.model = joblib.load(self.local_path)
        self.source = "local_joblib"
        logger.info("Successfully loaded %s from local artifact %s", self.model_name, self.local_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @abstractmethod
    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference and returns structured response."""
        pass
