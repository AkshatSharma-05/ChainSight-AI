"""Automated test suite for ChainSight-AI FastAPI application."""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ----------------- System Endpoint Tests -----------------

def test_root_endpoint():
    """Verifies that the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "ChainSight-AI"
    assert "version" in data
    assert "/health" in data["health_url"]
    assert "/docs" in data["docs_url"]

def test_health_endpoint():
    """Verifies health check and that all four models are reported as loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["demand_forecasting"] is True
    assert data["models_loaded"]["stockout_prediction"] is True
    assert data["models_loaded"]["delivery_eta"] is True
    assert data["models_loaded"]["seller_risk"] is True

def test_models_endpoint():
    """Verifies that all 4 models are listed with proper metadata."""
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 4
    model_ids = {m["id"] for m in models}
    assert model_ids == {"demand_forecasting", "stockout_prediction", "delivery_eta", "seller_risk"}

# ----------------- Model Prediction Tests -----------------

def test_predict_demand_valid():
    """Verifies valid demand forecasting inference."""
    payload = {
        "wday": 1, "month": 5, "year": 2016, "sell_price": 8.50,
        "lag_7": 12.0, "lag_14": 10.0, "lag_28": 15.0,
        "rolling_mean_7": 11.2, "rolling_mean_28": 13.5, "rolling_std_7": 2.1,
        "dept_id": "FOODS_1", "cat_id": "FOODS"
    }
    response = client.post("/predict/demand", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "DemandForecastingModel"
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] >= 0.0
    assert data["unit"] == "units"
    assert "latency_ms" in data

def test_predict_stockout_valid():
    """Verifies valid stockout prediction inference and output structure."""
    payload = {
        "opening_stock": 50, "received_stock": 10, "avg_daily_demand_7d": 8.0,
        "avg_daily_demand_30d": 7.5, "demand_std_7d": 1.5, "inventory_coverage_days": 6.25,
        "supplier_avg_lead_time": 5, "supplier_reliability": 0.95, "warehouse_utilization_pct": 75.0,
        "warehouse_id": "WH_001", "product_category": "Electronics", "day_of_week": 2, "month": 5
    }
    response = client.post("/predict/stockout", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "StockoutPredictionModel"
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["stockout_probability"] <= 1.0
    assert data["risk_severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "calibrated_threshold" in data

def test_predict_delivery_eta_valid():
    """Verifies valid delivery ETA inference."""
    payload = {
        "price": 45.0, "freight_value": 12.5, "product_weight_g": 500.0,
        "product_category_name": "beleza_saude", "seller_state": "SP",
        "customer_state": "RJ", "seller_delivery_count": 15
    }
    response = client.post("/predict/delivery-eta", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "DeliveryETAModel"
    assert data["prediction"] > 0
    assert data["unit"] == "days"

def test_predict_seller_risk_valid():
    """Verifies valid seller delay risk inference."""
    payload = {
        "price": 45.0, "freight_value": 12.5, "product_weight_g": 500.0,
        "product_category_name": "beleza_saude", "seller_state": "SP",
        "customer_state": "RJ", "seller_delivery_count": 15
    }
    response = client.post("/predict/seller-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "SellerRiskModel"
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["delay_probability"] <= 1.0
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")

# ----------------- Inventory Optimization Endpoint Tests -----------------

def test_inventory_recommendation():
    """Verifies inventory replenishment recommendation calculations."""
    payload = {
        "current_stock": 35,
        "unit_price": 25.0,
        "predicted_7d_demand": 85.0,
        "stockout_prob": 0.65,
        "delivery_eta_days": 6.5,
        "seller_delayed_prob": 0.40,
        "daily_demand_std": 2.5
    }
    response = client.post("/inventory/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimization_metrics" in data
    assert "recommendation" in data
    assert data["optimization_metrics"]["recommended_reorder_qty"] > 0
    assert data["recommendation"]["action"] in ("PLACE_PURCHASE_ORDER", "MAINTAIN_STOCK")

# ----------------- Digital Twin Disruption Simulation Tests -----------------

def test_digital_twin_disruption_simulation():
    """Verifies supply chain disruption simulation and strategy benchmarking."""
    payload = {
        "current_inventory": 100,
        "daily_demand": 20.0,
        "supplier_lead_time_days": 5,
        "unit_price": 50.0,
        "disruption_lead_time_delay_days": 6,
        "disruption_demand_surge_pct": 50.0,
        "lost_sale_penalty_multiplier": 1.5
    }
    response = client.post("/simulation/disrupt", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "simulation_scenario" in data
    assert "strategies_comparison" in data
    assert len(data["strategies_comparison"]) == 4
    assert "recommended_recovery_action" in data

# ----------------- Monitoring & Drift Tests -----------------

def test_monitoring_endpoints():
    """Verifies monitoring status and metrics tracking endpoints."""
    # Status endpoint
    r_status = client.get("/monitoring/status")
    assert r_status.status_code == 200
    data_status = r_status.json()
    assert "monitoring_status" in data_status
    assert "feature_drift_analysis" in data_status

    # Metrics endpoint
    r_metrics = client.get("/monitoring/metrics")
    assert r_metrics.status_code == 200
    data_metrics = r_metrics.json()
    assert "total_inferences" in data_metrics

# ----------------- Negative & Edge Case Tests -----------------

def test_predict_demand_invalid_category():
    """Verifies that an unknown department raises a validation error."""
    payload = {
        "wday": 1, "month": 5, "year": 2016, "sell_price": 8.50,
        "lag_7": 12.0, "lag_14": 10.0, "lag_28": 15.0,
        "rolling_mean_7": 11.2, "rolling_mean_28": 13.5, "rolling_std_7": 2.1,
        "dept_id": "UNKNOWN_DEPT", "cat_id": "FOODS"
    }
    response = client.post("/predict/demand", json=payload)
    assert response.status_code == 400
    assert "Invalid dept_id" in response.json()["detail"]

def test_predict_demand_missing_feature():
    """Verifies that missing required fields trigger HTTP 422 Unprocessable Entity."""
    payload = {
        "wday": 1, "month": 5,
    }
    response = client.post("/predict/demand", json=payload)
    assert response.status_code == 422

def test_predict_stockout_negative_inventory():
    """Verifies that negative opening stock triggers Pydantic schema validation error."""
    payload = {
        "opening_stock": -10, "received_stock": 10, "avg_daily_demand_7d": 8.0,
        "avg_daily_demand_30d": 7.5, "demand_std_7d": 1.5, "inventory_coverage_days": 6.25,
        "supplier_avg_lead_time": 5, "supplier_reliability": 0.95, "warehouse_utilization_pct": 75.0,
        "warehouse_id": "WH_001", "product_category": "Electronics", "day_of_week": 2, "month": 5
    }
    response = client.post("/predict/stockout", json=payload)
    assert response.status_code == 422

def test_predict_eta_incorrect_type():
    """Verifies that string value for numerical price triggers HTTP 422."""
    payload = {
        "price": "not_a_number", "freight_value": 12.5, "product_weight_g": 500.0,
        "product_category_name": "beleza_saude", "seller_state": "SP",
        "customer_state": "RJ", "seller_delivery_count": 15
    }
    response = client.post("/predict/delivery-eta", json=payload)
    assert response.status_code == 422
