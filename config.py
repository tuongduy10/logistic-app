from dataclasses import dataclass, field
import yaml
import os

# ===== LOAD YAML CONFIG =====
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_PATH, "r") as f:
    _yaml_config = yaml.safe_load(f)

# ===== DYNAMIC CONFIG FROM YAML =====
FORECAST_HORIZON = _yaml_config.get("forecast_horizon", 12)
INPUT_FILE = _yaml_config.get("input_file", "input_3rd.xlsx")
OUTPUT_CHARTS_DIR = "output_charts"
OUTPUT_FILES_DIR = "output_files"

COLORS: dict[str, str] = {
    "history":      "#4C72B0",
    "forecast":     "#DD8452",
    "band":         "#DD8452",
    "test_pred":    "#C44E52",
    "ARIMA":        "#4C72B0",
    "Prophet":      "#55A868",
    "RandomForest": "#C44E52",
    "high":         "#C44E52",
    "medium":       "#DD8452",
    "low":          "#4C72B0",
    "stable":       "#55A868",
    "volatile":     "#C44E52",
}

MPL_THEME: dict = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
}

# Default inventory parameters when SKU is not found in parameters sheet
DEFAULT_INVENTORY_PARAMS: dict = {
    "lead_time_days_mean":          60.0,
    "ordering_cost_usd_per_order":  50_000.0,
    "holding_cost_usd_per_unit_year": 2_000.0,
    "z_value":                      1.65,
}

# Column aliases for flexible input column name mapping
COLUMN_ALIASES: dict[str, list[str]] = {
    "date":     ["date", "month", "period", "ds"],
    "sku_id":   ["sku_id", "sku", "item_id", "product_id", "code"],
    "sku_name": ["sku_name", "item_name", "product_name", "name"],
    "demand":   [
        "demand", "qty", "quantity", "sales", "forecast_qty",
        "actual_demand_units", "demand_units", "units_sold", "y",
    ],
}
