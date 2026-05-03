"""
InventoryCalculator: EOQ / Safety Stock / ROP computation.
Pure functions — no side effects, easy to unit-test.
"""
import numpy as np

from core.models import ForecastResult, InventoryParams, InventoryResult


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = np.mean(values)
    return round(float(np.std(values) / mean), 3) if mean > 0 else 0.0


def variability_label(cv: float) -> str:
    """CV < 0.3 → Stable; CV >= 0.3 → Volatile (supply-chain standard)."""
    return "Stable" if cv < 0.3 else "Volatile"


def demand_level_label(annual_demand: float, p33: float, p67: float) -> str:
    """Percentile-based classification — scale-independent."""
    if annual_demand >= p67:
        return "High"
    if annual_demand >= p33:
        return "Medium"
    return "Low"


class InventoryCalculator:
    def compute(
        self,
        forecast: ForecastResult,
        params: InventoryParams,
    ) -> InventoryResult:
        fc = forecast.values
        demand_mean = float(np.mean(fc))
        demand_std = float(np.std(fc))
        annual_demand = demand_mean * 12
        lead_time_months = params.lead_time_days / 30.0

        eoq = np.sqrt(
            (2 * annual_demand * params.order_cost) / params.holding_cost
        )
        safety_stock = (
            params.service_level_z * demand_std * np.sqrt(lead_time_months)
        )
        rop = demand_mean * lead_time_months + safety_stock
        total_cost = (
            (annual_demand / eoq) * params.order_cost
            + (eoq / 2) * params.holding_cost
        )

        cv = coefficient_of_variation(fc)

        return InventoryResult(
            sku_id=forecast.sku_id,
            sku_name=forecast.sku_name,
            annual_demand=round(annual_demand, 2),
            demand_mean_monthly=round(demand_mean, 2),
            demand_std_monthly=round(demand_std, 2),
            cv=cv,
            safety_stock=round(safety_stock, 2),
            rop=round(rop, 2),
            eoq=round(eoq, 2),
            total_cost=round(total_cost, 2),
            variability=variability_label(cv),
        )
