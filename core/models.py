from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class InventoryParams:
    lead_time_days: float
    order_cost: float
    holding_cost: float
    service_level_z: float


@dataclass
class ModelEvalResult:
    sku_id: str
    sku_name: str
    mape_scores: dict[str, float]          # {"ARIMA": 0.05, "Prophet": 0.08, ...}
    best_model: str
    test_predictions: np.ndarray


@dataclass
class ForecastResult:
    sku_id: str
    sku_name: str
    dates: list[pd.Timestamp]
    values: np.ndarray
    model_used: str


@dataclass
class InventoryResult:
    sku_id: str
    sku_name: str
    annual_demand: float
    demand_mean_monthly: float
    demand_std_monthly: float
    cv: float
    safety_stock: float
    rop: float
    eoq: float
    total_cost: float
    variability: str
    demand_level: str = ""


@dataclass
class SKUProcessingContext:
    """Bundles all data for a single SKU to pass between pipeline stages."""
    sku_id: str
    sku_name: str
    ts: pd.Series
    train: pd.Series
    test: pd.Series
    inventory_params: InventoryParams
    eval_result: ModelEvalResult | None = None
    forecast_result: ForecastResult | None = None
    inventory_result: InventoryResult | None = None
