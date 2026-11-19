"""Supply Chain Digital Twin & Disruption Simulation Engine.

Simulates supply chain disruptions (demand surges, shipping delays, warehouse stock shocks)
and benchmarks recovery strategies to recommend optimal resilience actions.
"""
import math
from typing import Any, Dict, List

class SupplyChainDigitalTwin:
    """Simulates disruption scenarios and recovery action economics."""

    def simulate_disruption(
        self,
        current_inventory: int,
        daily_demand: float,
        supplier_lead_time_days: int,
        unit_price: float,
        disruption_lead_time_delay_days: int = 5,
        disruption_demand_surge_pct: float = 40.0,
        lost_sale_penalty_multiplier: float = 1.5,
    ) -> Dict[str, Any]:
        """Simulates disruption and evaluates 4 candidate recovery strategies."""
        # 1. Baseline Simulation (Under Disruption)
        disrupted_daily_demand = daily_demand * (1.0 + (disruption_demand_surge_pct / 100.0))
        disrupted_lead_time = supplier_lead_time_days + disruption_lead_time_delay_days

        # Total demand expected before replenishment arrives
        expected_demand_during_disruption = math.ceil(disrupted_daily_demand * disrupted_lead_time)

        # Inventory balance calculation
        shortage_units = max(0, expected_demand_during_disruption - current_inventory)
        days_until_stockout = max(0.0, round(current_inventory / disrupted_daily_demand, 1)) if disrupted_daily_demand > 0 else 999.0

        lost_sale_cost_per_unit = unit_price * lost_sale_penalty_multiplier

        # 2. Evaluate Alternative Recovery Strategies
        strategies: List[Dict[str, Any]] = []

        # Strategy A: Baseline - Do Nothing (Wait for Delayed Supplier)
        cost_a = shortage_units * lost_sale_cost_per_unit
        strategies.append({
            "strategy_id": "WAIT_AND_MAINTAIN",
            "name": "Continue with Existing Supplier",
            "description": "No operational changes. Absorb supplier delay and lost sales.",
            "operational_delay_days": disruption_lead_time_delay_days,
            "projected_lost_sales_units": shortage_units,
            "intervention_cost": 0.0,
            "total_estimated_disruption_cost": round(cost_a, 2),
            "stockout_prevented": shortage_units == 0
        })

        # Strategy B: Alternative Local Supplier (2-day lead time, 25% price premium)
        alt_lead_time = 2
        alt_premium_per_unit = unit_price * 0.25
        alt_order_qty = max(10, shortage_units)
        alt_shortage = max(0, math.ceil(disrupted_daily_demand * alt_lead_time) - current_inventory)
        cost_b = (alt_order_qty * alt_premium_per_unit) + (alt_shortage * lost_sale_cost_per_unit)
        strategies.append({
            "strategy_id": "ALTERNATIVE_SUPPLIER",
            "name": "Order from Alternative Supplier",
            "description": "Engage secondary certified supplier at 25% premium with 2-day delivery.",
            "operational_delay_days": 2,
            "projected_lost_sales_units": alt_shortage,
            "intervention_cost": round(alt_order_qty * alt_premium_per_unit, 2),
            "total_estimated_disruption_cost": round(cost_b, 2),
            "stockout_prevented": alt_shortage == 0
        })

        # Strategy C: Expedited Air Freight (Cut delay by 70%, fixed $250 freight fee)
        expedited_lead_time = max(1, supplier_lead_time_days + math.ceil(disruption_lead_time_delay_days * 0.3))
        expedited_shortage = max(0, math.ceil(disrupted_daily_demand * expedited_lead_time) - current_inventory)
        freight_fee = 250.0
        cost_c = freight_fee + (expedited_shortage * lost_sale_cost_per_unit)
        strategies.append({
            "strategy_id": "EXPEDITED_FREIGHT",
            "name": "Expedited Air Shipping",
            "description": "Upgrade in-transit shipment to priority air courier.",
            "operational_delay_days": math.ceil(disruption_lead_time_delay_days * 0.3),
            "projected_lost_sales_units": expedited_shortage,
            "intervention_cost": freight_fee,
            "total_estimated_disruption_cost": round(cost_c, 2),
            "stockout_prevented": expedited_shortage == 0
        })

        # Strategy D: Emergency Cross-Dock Warehouse Transfer (Fixed transfer fee $120, 1-day delivery)
        transfer_units = min(current_inventory + 50, shortage_units)
        transfer_shortage = max(0, shortage_units - transfer_units)
        transfer_fee = 120.0
        cost_d = transfer_fee + (transfer_shortage * lost_sale_cost_per_unit)
        strategies.append({
            "strategy_id": "WAREHOUSE_TRANSFER",
            "name": "Inter-Warehouse Transfer",
            "description": "Transfer buffer inventory from nearest regional fulfillment center.",
            "operational_delay_days": 1,
            "projected_lost_sales_units": transfer_shortage,
            "intervention_cost": transfer_fee,
            "total_estimated_disruption_cost": round(cost_d, 2),
            "stockout_prevented": transfer_shortage == 0
        })

        # Select strategy with minimum total disruption cost
        best_strategy = min(strategies, key=lambda s: s["total_estimated_disruption_cost"])

        return {
            "simulation_scenario": {
                "supplier_delay_days": disruption_lead_time_delay_days,
                "demand_surge_percentage": disruption_demand_surge_pct,
                "disrupted_daily_demand": round(disrupted_daily_demand, 2),
                "total_effective_lead_time_days": disrupted_lead_time,
                "days_until_stockout": days_until_stockout,
                "unmitigated_shortage_units": shortage_units
            },
            "strategies_comparison": strategies,
            "recommended_recovery_action": {
                "recommended_strategy_id": best_strategy["strategy_id"],
                "name": best_strategy["name"],
                "reason": f"Minimizes total disruption cost to ${best_strategy['total_estimated_disruption_cost']:.2f}.",
                "cost_saving_vs_do_nothing": round(max(0.0, cost_a - best_strategy["total_estimated_disruption_cost"]), 2)
            }
        }
