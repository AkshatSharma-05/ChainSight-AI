"""Inventory Optimization Engine for ChainSight-AI.

Combines ML predictions (Demand, Stockout, Delivery ETA, Seller Risk)
with standard Operations Research formulas (Dynamic Reorder Point & Safety Stock)
to generate transparent replenishment recommendations.
"""
import math
from typing import Any, Dict

class InventoryOptimizationEngine:
    """Calculates optimal reorder quantities and replenishment schedules."""

    def __init__(self, service_level_z: float = 1.65, holding_cost_pct: float = 0.20, order_cost: float = 50.0):
        # Default Z = 1.65 corresponds to a 95% service level cycle
        self.service_level_z = service_level_z
        self.holding_cost_pct = holding_cost_pct
        self.order_cost = order_cost

    def recommend(
        self,
        current_stock: int,
        unit_price: float,
        predicted_7d_demand: float,
        stockout_prob: float,
        delivery_eta_days: float,
        seller_delayed_prob: float,
        daily_demand_std: float = 2.0,
    ) -> Dict[str, Any]:
        """Generates inventory replenishment decision based on ML predictions and OR formulas."""
        # 1. Base Rates from ML Predictions
        avg_daily_demand = max(0.1, predicted_7d_demand / 7.0)
        base_lead_time_days = max(1.0, delivery_eta_days)

        # 2. Business Rules: Risk-Adjusted Lead Time Buffer
        # If seller risk is elevated or stockout risk is high, inflate safety buffer
        lead_time_risk_buffer = 1.0 + (seller_delayed_prob * 0.5)
        effective_lead_time = base_lead_time_days * lead_time_risk_buffer

        # 3. Operations Research Formulation:
        # Dynamic Safety Stock = Z * sigma_L
        # where sigma_L = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        # Simplified standard supply chain approximation: SS = Z * sigma_d * sqrt(L)
        safety_stock = math.ceil(self.service_level_z * daily_demand_std * math.sqrt(effective_lead_time))

        # Reorder Point (ROP) = (Daily Demand * Lead Time) + Safety Stock
        reorder_point = math.ceil((avg_daily_demand * effective_lead_time) + safety_stock)

        # Economic Order Quantity (EOQ): sqrt( (2 * D_annual * S) / H )
        annual_demand = avg_daily_demand * 365.0
        annual_holding_cost_per_unit = max(1.0, unit_price * self.holding_cost_pct)
        raw_eoq = math.sqrt((2.0 * annual_demand * self.order_cost) / annual_holding_cost_per_unit)
        eoq = max(10, math.ceil(raw_eoq))

        # Reorder Decision
        needs_reorder = (current_stock <= reorder_point) or (stockout_prob >= 0.4556)

        if needs_reorder:
            deficit = max(0, reorder_point - current_stock)
            # Order at least EOQ or the deficit, whichever covers the replenishment cycle
            recommended_reorder_qty = max(eoq, deficit + math.ceil(avg_daily_demand * 7))
            action = "PLACE_PURCHASE_ORDER"
            action_summary = f"Stock level ({current_stock}) is at or below ROP ({reorder_point}). Order {recommended_reorder_qty} units immediately."
        else:
            recommended_reorder_qty = 0
            action = "MAINTAIN_STOCK"
            action_summary = f"Inventory level ({current_stock}) is adequate above ROP ({reorder_point}). No reorder necessary."

        # Estimated Financials
        estimated_order_value = round(recommended_reorder_qty * unit_price, 2)
        estimated_annual_holding_cost = round(current_stock * annual_holding_cost_per_unit, 2)

        return {
            "ml_inputs": {
                "predicted_7d_demand_units": round(predicted_7d_demand, 2),
                "stockout_probability": round(stockout_prob, 4),
                "delivery_eta_days": round(delivery_eta_days, 1),
                "seller_delay_probability": round(seller_delayed_prob, 4)
            },
            "business_rules": {
                "service_level_target": "95%",
                "holding_cost_rate": f"{int(self.holding_cost_pct * 100)}%",
                "lead_time_risk_multiplier": round(lead_time_risk_buffer, 2)
            },
            "optimization_metrics": {
                "current_stock": current_stock,
                "safety_stock_units": safety_stock,
                "reorder_point_units": reorder_point,
                "economic_order_qty_eoq": eoq,
                "recommended_reorder_qty": recommended_reorder_qty
            },
            "financial_estimates": {
                "unit_price": unit_price,
                "estimated_order_cost": estimated_order_value,
                "estimated_annual_holding_cost": estimated_annual_holding_cost
            },
            "recommendation": {
                "action": action,
                "summary": action_summary,
                "urgency": "HIGH" if stockout_prob >= 0.50 else ("MEDIUM" if needs_reorder else "LOW")
            }
        }
