"""Pydantic schemas for ChainSight-AI API."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ----------------- Prediction Requests -----------------

class DemandRequest(BaseModel):
    wday: int = Field(..., ge=1, le=7, description="Day of week (1=Saturday ... 7=Friday)", examples=[1])
    month: int = Field(..., ge=1, le=12, description="Month (1-12)", examples=[5])
    year: int = Field(..., ge=2010, le=2030, description="Calendar year", examples=[2016])
    sell_price: float = Field(..., gt=0, description="Unit sell price", examples=[8.50])
    lag_7: float = Field(..., ge=0, description="Sales 7 days ago", examples=[12.0])
    lag_14: float = Field(..., ge=0, description="Sales 14 days ago", examples=[10.0])
    lag_28: float = Field(..., ge=0, description="Sales 28 days ago", examples=[15.0])
    rolling_mean_7: float = Field(..., ge=0, description="7-day rolling mean sales", examples=[11.2])
    rolling_mean_28: float = Field(..., ge=0, description="28-day rolling mean sales", examples=[13.5])
    rolling_std_7: float = Field(..., ge=0, description="7-day rolling std of sales", examples=[2.1])
    dept_id: str = Field(..., description="Department ID (e.g. FOODS_1, HOBBIES_1)", examples=["FOODS_1"])
    cat_id: str = Field(..., description="Category ID (e.g. FOODS, HOBBIES, HOUSEHOLD)", examples=["FOODS"])

class StockoutRequest(BaseModel):
    opening_stock: int = Field(..., ge=0, description="Current stock in warehouse", examples=[50])
    received_stock: int = Field(..., ge=0, description="Recently received stock", examples=[10])
    avg_daily_demand_7d: float = Field(..., ge=0, description="7-day average daily demand", examples=[8.0])
    avg_daily_demand_30d: float = Field(..., ge=0, description="30-day average daily demand", examples=[7.5])
    demand_std_7d: float = Field(..., ge=0, description="7-day demand standard deviation", examples=[1.5])
    inventory_coverage_days: float = Field(..., ge=0, description="Estimated coverage in days", examples=[6.25])
    supplier_avg_lead_time: int = Field(..., ge=1, description="Lead time in days", examples=[5])
    supplier_reliability: float = Field(..., ge=0.0, le=1.0, description="Supplier on-time rate (0 to 1)", examples=[0.95])
    warehouse_utilization_pct: float = Field(..., ge=0.0, le=100.0, description="Warehouse capacity used %", examples=[75.0])
    warehouse_id: str = Field(..., description="Warehouse ID (WH_001 to WH_005)", examples=["WH_001"])
    product_category: str = Field(..., description="Product category name", examples=["Electronics"])
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday ... 6=Sunday)", examples=[2])
    month: int = Field(..., ge=1, le=12, description="Month (1-12)", examples=[5])

class DeliveryETARRequest(BaseModel):
    price: float = Field(..., gt=0, description="Product price in BRL", examples=[45.0])
    freight_value: float = Field(..., ge=0, description="Shipping cost in BRL", examples=[12.5])
    product_weight_g: float = Field(..., gt=0, description="Product weight in grams", examples=[500.0])
    product_category_name: str = Field(..., description="Olist category name", examples=["beleza_saude"])
    seller_state: str = Field(..., min_length=2, max_length=2, description="2-letter seller state code", examples=["SP"])
    customer_state: str = Field(..., min_length=2, max_length=2, description="2-letter customer state code", examples=["RJ"])
    seller_delivery_count: int = Field(..., ge=0, description="Prior order deliveries by seller", examples=[15])

class SellerRiskRequest(BaseModel):
    price: float = Field(..., gt=0, description="Product price in BRL", examples=[45.0])
    freight_value: float = Field(..., ge=0, description="Shipping cost in BRL", examples=[12.5])
    product_weight_g: float = Field(..., gt=0, description="Product weight in grams", examples=[500.0])
    product_category_name: str = Field(..., description="Olist category name", examples=["beleza_saude"])
    seller_state: str = Field(..., min_length=2, max_length=2, description="2-letter seller state code", examples=["SP"])
    customer_state: str = Field(..., min_length=2, max_length=2, description="2-letter customer state code", examples=["RJ"])
    seller_delivery_count: int = Field(..., ge=0, description="Prior order deliveries by seller", examples=[15])

# ----------------- Inventory Optimization Request -----------------

class InventoryRecommendRequest(BaseModel):
    current_stock: int = Field(..., ge=0, description="Current on-hand warehouse inventory units", examples=[40])
    unit_price: float = Field(..., gt=0, description="Product unit price", examples=[25.0])
    predicted_7d_demand: float = Field(..., ge=0, description="Forecasted 7-day demand from ML model", examples=[85.0])
    stockout_prob: float = Field(..., ge=0.0, le=1.0, description="Stockout probability from ML model", examples=[0.65])
    delivery_eta_days: float = Field(..., gt=0, description="Supplier delivery ETA in days from ML model", examples=[6.5])
    seller_delayed_prob: float = Field(..., ge=0.0, le=1.0, description="Seller delay probability from ML model", examples=[0.40])
    daily_demand_std: float = Field(default=2.5, ge=0.1, description="Standard deviation of daily demand", examples=[2.5])

# ----------------- Digital Twin Disruption Request -----------------

class DisruptionSimulationRequest(BaseModel):
    current_inventory: int = Field(..., ge=0, description="Current stock units in warehouse", examples=[120])
    daily_demand: float = Field(..., gt=0, description="Normal baseline daily demand units", examples=[25.0])
    supplier_lead_time_days: int = Field(..., ge=1, description="Normal baseline supplier lead time days", examples=[7])
    unit_price: float = Field(..., gt=0, description="Product unit price", examples=[45.0])
    disruption_lead_time_delay_days: int = Field(default=5, ge=0, description="Simulated shipping delay in days", examples=[5])
    disruption_demand_surge_pct: float = Field(default=40.0, ge=0.0, description="Simulated demand spike percentage", examples=[40.0])
    lost_sale_penalty_multiplier: float = Field(default=1.5, ge=1.0, description="Cost multiplier for lost sales/stockout", examples=[1.5])

# ----------------- System & General Responses -----------------

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    task: str
    algorithm: str
    features: List[str]
    is_loaded: bool
    source: str
