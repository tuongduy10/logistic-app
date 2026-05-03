"""
Repository Pattern: all file I/O is isolated here.
Swap Excel → database → API without touching any business logic.
"""
from __future__ import annotations

import pandas as pd

from config import COLUMN_ALIASES, DEFAULT_INVENTORY_PARAMS
from core.models import InventoryParams


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: str(col).strip().lower().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=renamed)
    for target, candidates in COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        for cand in candidates:
            if cand in df.columns:
                df = df.rename(columns={cand: target})
                break
    return df


class SalesRepository:
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        raw = pd.read_excel(self._file_path, sheet_name="sales_history")
        df = _normalize_columns(raw)

        required = {"date", "sku_id", "sku_name", "demand"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in sales_history: {sorted(missing)}")

        df["date"] = pd.to_datetime(df["date"])
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
        df = df.dropna(subset=["date", "sku_id", "sku_name", "demand"])

        self._df = df
        return df

    def sku_ids(self) -> list:
        return self.load()["sku_id"].unique().tolist()

    def get_sku_series(self, sku_id) -> tuple[pd.Series, pd.Series, pd.Series, str]:
        """Returns (ts, train, test, sku_name) for a given SKU."""
        df = self.load()
        data = df[df["sku_id"] == sku_id].sort_values("date")
        sku_name = data["sku_name"].iloc[0]
        indexed = data[["date", "demand"]].set_index("date")["demand"]
        split = data[["date", "train_test_split"]].set_index("date")["train_test_split"]
        train = indexed[split == "train"]
        test = indexed[split == "test"]
        return indexed, train, test, sku_name


class InventoryParamsRepository:
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._params: dict | None = None

    def load(self) -> dict[str, InventoryParams]:
        if self._params is not None:
            return self._params

        raw = pd.read_excel(self._file_path, sheet_name="inventory_parameters")
        df = _normalize_columns(raw)

        required = {
            "sku_id", "lead_time_days_mean",
            "ordering_cost_usd_per_order",
            "holding_cost_usd_per_unit_year",
            "z_value",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in inventory_parameters: {sorted(missing)}")

        self._params = {
            row["sku_id"]: InventoryParams(
                lead_time_days=float(row["lead_time_days_mean"]),
                order_cost=float(row["ordering_cost_usd_per_order"]),
                holding_cost=float(row["holding_cost_usd_per_unit_year"]),
                service_level_z=float(row["z_value"]),
            )
            for _, row in df.iterrows()
        }
        return self._params

    def get(self, sku_id) -> InventoryParams:
        params = self.load()
        if sku_id in params:
            return params[sku_id]
        d = DEFAULT_INVENTORY_PARAMS
        return InventoryParams(
            lead_time_days=d["lead_time_days_mean"],
            order_cost=d["ordering_cost_usd_per_order"],
            holding_cost=d["holding_cost_usd_per_unit_year"],
            service_level_z=d["z_value"],
        )


class ExcelExporter:
    def export(
        self,
        output_path: str,
        forecast_rows: list[dict],
        eval_rows: list[dict],
        inventory_rows: list[dict],
    ) -> None:
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame(forecast_rows).to_excel(
                writer, sheet_name="forecast_output", index=False
            )
            pd.DataFrame(eval_rows).to_excel(
                writer, sheet_name="model_evaluation", index=False
            )
            pd.DataFrame(inventory_rows).to_excel(
                writer, sheet_name="inventory_output", index=False
            )
        print(f"✓ Excel saved: {output_path}")

    def export_inventory(self, output_path: str, inventory_rows: list[dict]) -> None:
        """Xuất inventory_output ra file riêng biệt."""
        pd.DataFrame(inventory_rows).to_excel(output_path, index=False)
        print(f"✓ Inventory output saved: {output_path}")
