"""API route definitions for ChainSight-AI."""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from src.api.schemas import (
    DemandRequest,
    StockoutRequest,
    DeliveryETARRequest,
    SellerRiskRequest,
    InventoryRecommendRequest,
    DisruptionSimulationRequest,
    HealthResponse,
    ModelInfo,
)
from src.services.demand_service import DemandForecastingService
from src.services.stockout_service import StockoutPredictionService
from src.services.eta_service import DeliveryETAService
from src.services.seller_service import SellerRiskService
from src.inventory.engine import InventoryOptimizationEngine
from src.simulation.digital_twin import SupplyChainDigitalTwin
from src.monitoring.drift_detector import monitor

logger = logging.getLogger("chainsight.api")

router = APIRouter()

# Initialize services
demand_service = DemandForecastingService()
stockout_service = StockoutPredictionService()
eta_service = DeliveryETAService()
seller_service = SellerRiskService()

# Initialize optimization and simulation engines
inventory_engine = InventoryOptimizationEngine()
digital_twin = SupplyChainDigitalTwin()

@router.get("/", tags=["System"])
def root() -> Dict[str, Any]:
    """Root endpoint welcoming user and providing API links."""
    return {
        "project": "ChainSight-AI",
        "title": "Autonomous Supply Chain Resilience & Decision Intelligence Platform",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": [
            "/predict/demand",
            "/predict/stockout",
            "/predict/delivery-eta",
            "/predict/seller-risk",
            "/inventory/recommend",
            "/simulation/disrupt",
            "/monitoring/status",
            "/monitoring/metrics"
        ]
    }

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """System health check reporting load status for all 4 models."""
    models_status = {
        "demand_forecasting": demand_service.is_loaded,
        "stockout_prediction": stockout_service.is_loaded,
        "delivery_eta": eta_service.is_loaded,
        "seller_risk": seller_service.is_loaded,
    }
    all_ok = all(models_status.values())
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        version="1.0.0",
        models_loaded=models_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

@router.get("/models", tags=["System"])
def list_models() -> List[Dict[str, Any]]:
    """Lists all available machine learning models and their metadata."""
    return [
        {
            "id": "demand_forecasting",
            "name": "DemandForecastingModel",
            "task": "Time-Series Tabular Regression",
            "algorithm": "LightGBMRegressor",
            "is_loaded": demand_service.is_loaded,
            "source": demand_service.source,
            "features_count": 12,
        },
        {
            "id": "stockout_prediction",
            "name": "StockoutPredictionModel",
            "task": "Binary Classification",
            "algorithm": "Pipeline(ColumnTransformer + LGBMClassifier)",
            "is_loaded": stockout_service.is_loaded,
            "source": stockout_service.source,
            "features_count": 13,
        },
        {
            "id": "delivery_eta",
            "name": "DeliveryETAModel",
            "task": "Regression",
            "algorithm": "CatBoostRegressor",
            "is_loaded": eta_service.is_loaded,
            "source": eta_service.source,
            "features_count": 7,
        },
        {
            "id": "seller_risk",
            "name": "SellerRiskModel",
            "task": "Binary Classification",
            "algorithm": "CatBoostClassifier",
            "is_loaded": seller_service.is_loaded,
            "source": seller_service.source,
            "features_count": 7,
        },
    ]

# ----------------- Prediction Endpoints -----------------

@router.post("/predict/demand", tags=["Predictions"])
def predict_demand(payload: DemandRequest) -> Dict[str, Any]:
    """Forecasts unit product demand for the next 7 days."""
    try:
        data = payload.model_dump()
        result = demand_service.predict(data)
        monitor.record_inference(
            model_name="DemandForecastingModel",
            prediction=result["prediction"],
            latency_ms=result["latency_ms"],
            features=data
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.error("Demand prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {exc}")

@router.post("/predict/stockout", tags=["Predictions"])
def predict_stockout(payload: StockoutRequest) -> Dict[str, Any]:
    """Predicts probability of a warehouse stockout over the next 7 days."""
    try:
        data = payload.model_dump()
        result = stockout_service.predict(data)
        monitor.record_inference(
            model_name="StockoutPredictionModel",
            prediction=result["prediction"],
            latency_ms=result["latency_ms"],
            features=data
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.error("Stockout prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {exc}")

@router.post("/predict/delivery-eta", tags=["Predictions"])
def predict_delivery_eta(payload: DeliveryETARRequest) -> Dict[str, Any]:
    """Predicts delivery transit ETA duration in days."""
    try:
        data = payload.model_dump()
        result = eta_service.predict(data)
        monitor.record_inference(
            model_name="DeliveryETAModel",
            prediction=result["prediction"],
            latency_ms=result["latency_ms"],
            features=data
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.error("Delivery ETA prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {exc}")

@router.post("/predict/seller-risk", tags=["Predictions"])
def predict_seller_risk(payload: SellerRiskRequest) -> Dict[str, Any]:
    """Predicts shipping delay risk associated with an order and seller."""
    try:
        data = payload.model_dump()
        result = seller_service.predict(data)
        monitor.record_inference(
            model_name="SellerRiskModel",
            prediction=result["prediction"],
            latency_ms=result["latency_ms"],
            features=data
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.error("Seller risk prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {exc}")

# ----------------- Inventory Optimization Endpoint -----------------

@router.post("/inventory/recommend", tags=["Inventory Optimization"])
def recommend_inventory(payload: InventoryRecommendRequest) -> Dict[str, Any]:
    """Generates replenishment and safety stock recommendations combining all 4 ML models."""
    try:
        return inventory_engine.recommend(
            current_stock=payload.current_stock,
            unit_price=payload.unit_price,
            predicted_7d_demand=payload.predicted_7d_demand,
            stockout_prob=payload.stockout_prob,
            delivery_eta_days=payload.delivery_eta_days,
            seller_delayed_prob=payload.seller_delayed_prob,
            daily_demand_std=payload.daily_demand_std,
        )
    except Exception as exc:
        logger.error("Inventory recommendation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

# ----------------- Digital Twin Disruption Simulation Endpoint -----------------

@router.post("/simulation/disrupt", tags=["Digital Twin"])
def simulate_disruption(payload: DisruptionSimulationRequest) -> Dict[str, Any]:
    """Simulates supply chain disruptions and benchmarks alternative recovery actions."""
    try:
        return digital_twin.simulate_disruption(
            current_inventory=payload.current_inventory,
            daily_demand=payload.daily_demand,
            supplier_lead_time_days=payload.supplier_lead_time_days,
            unit_price=payload.unit_price,
            disruption_lead_time_delay_days=payload.disruption_lead_time_delay_days,
            disruption_demand_surge_pct=payload.disruption_demand_surge_pct,
            lost_sale_penalty_multiplier=payload.lost_sale_penalty_multiplier,
        )
    except Exception as exc:
        logger.error("Disruption simulation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

# ----------------- Monitoring & Drift Detection Endpoints -----------------

@router.get("/monitoring/status", tags=["Monitoring"])
def get_monitoring_status() -> Dict[str, Any]:
    """Reports data drift status comparing live inference data to training baselines."""
    return monitor.detect_drift()

@router.get("/monitoring/metrics", tags=["Monitoring"])
def get_monitoring_metrics() -> Dict[str, Any]:
    """Returns inference throughput and latency performance statistics."""
    return monitor.get_system_metrics()
