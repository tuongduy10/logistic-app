"""
InventoryFromForecastPipeline:
Independent stage that consumes the previously-exported `forecast_output`
sheet as the source of truth, recomputes inventory metrics, and writes a
standalone `inventory_output_<timestamp>.xlsx` file.

This decouples inventory calculation from the in-memory ForecastResult,
making the forecast Excel sheet a real contract between the two stages.

Two extension points:
  - inventory_params_file: override inventory parameters source
  - demand_adjuster:       apply business policy to forecast values
                           (e.g. +10% safety buffer for Q4)
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from config import INPUT_FILE, OUTPUT_FILES_DIR
from core.models import ForecastResult, InventoryResult
from core.repository import InventoryParamsRepository
from models.demand_adjusters import IDemandAdjuster, NoAdjustment
from models.inventory import InventoryCalculator, demand_level_label


class ForecastSheetReader:
    """Repository for the exported `forecast_output` sheet."""

    REQUIRED_COLS = {"date", "sku_id", "sku_name", "forecast_demand", "model_used"}

    def __init__(self, file_path: str, sheet_name: str = "forecast_output"):
        self._file_path = file_path
        self._sheet_name = sheet_name

    @property
    def file_path(self) -> str:
        return self._file_path

    def load(self) -> pd.DataFrame:
        df = pd.read_excel(self._file_path, sheet_name=self._sheet_name)
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in '{self._sheet_name}': {sorted(missing)}"
            )
        df["date"] = pd.to_datetime(df["date"])
        df["forecast_demand"] = pd.to_numeric(df["forecast_demand"], errors="coerce")
        return df.dropna(subset=["date", "sku_id", "forecast_demand"])

    def to_forecast_results(self) -> list[ForecastResult]:
        """Group rows by SKU and rebuild ForecastResult objects."""
        df = self.load().sort_values(["sku_id", "date"])
        results: list[ForecastResult] = []
        for sku_id, grp in df.groupby("sku_id", sort=False):
            results.append(
                ForecastResult(
                    sku_id=sku_id,
                    sku_name=str(grp["sku_name"].iloc[0]),
                    dates=list(grp["date"]),
                    values=grp["forecast_demand"].to_numpy(dtype=float),
                    model_used=str(grp["model_used"].iloc[0]),
                )
            )
        return results


class InventoryExcelExporter:
    """Writes a standalone inventory workbook (one sheet)."""

    def export(self, output_path: str, inventory_rows: Iterable[dict]) -> None:
        df = pd.DataFrame(list(inventory_rows))
        with pd.ExcelWriter(output_path) as writer:
            df.to_excel(writer, sheet_name="inventory_output", index=False)
        print(f"✓ Standalone inventory file saved: {output_path}")


class InventoryFromForecastPipeline:
    """
    Reads forecast_output sheet → applies demand adjustment policy
    → recomputes inventory → exports inventory_output_<timestamp>.xlsx.

    Args:
        forecast_file:          Excel file containing `forecast_output` sheet.
        inventory_params_file:  Excel file containing `inventory_parameters` sheet.
                                Defaults to project INPUT_FILE.
        demand_adjuster:        Strategy applied to forecast values before
                                inventory computation. Defaults to NoAdjustment.
        output_dir:             Where to write the standalone file.
        filename_suffix:        Extra suffix in output filename for traceability.
                                Defaults to the adjuster's name when policy != none.
    """

    def __init__(
        self,
        forecast_file: str,
        inventory_params_file: str = INPUT_FILE,
        demand_adjuster: IDemandAdjuster | None = None,
        output_dir: str = OUTPUT_FILES_DIR,
        filename_suffix: str | None = None,
    ):
        self._reader = ForecastSheetReader(forecast_file)
        self._inv_repo = InventoryParamsRepository(inventory_params_file)
        self._inv_calc = InventoryCalculator()
        self._exporter = InventoryExcelExporter()
        self._adjuster: IDemandAdjuster = demand_adjuster or NoAdjustment()
        self._output_dir = output_dir
        self._filename_suffix = filename_suffix
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> str:
        print(
            f"[InventoryFromForecast] forecast='{self._reader.file_path}' | "
            f"params='{self._inv_repo._file_path}' | "
            f"adjuster={self._adjuster.name}"
        )
        forecasts = self._reader.to_forecast_results()
        print(f"[InventoryFromForecast] Loaded {len(forecasts)} SKU forecasts.")

        # 1) Apply demand adjustment policy (Strategy)
        adjusted: list[ForecastResult] = [
            replace(fc, values=self._adjuster.adjust(fc.values)) for fc in forecasts
        ]

        # 2) Compute inventory metrics per SKU
        results: list[InventoryResult] = []
        for fc in adjusted:
            params = self._inv_repo.get(fc.sku_id)
            results.append(self._inv_calc.compute(fc, params))

        # 3) Cross-SKU demand level classification (33/67 percentile)
        rows = [vars(r) for r in results]
        annual = np.array([r["annual_demand"] for r in rows], dtype=float)
        if len(annual):
            p33, p67 = float(np.percentile(annual, 33)), float(np.percentile(annual, 67))
            for r in rows:
                r["demand_level"] = demand_level_label(r["annual_demand"], p33, p67)
            print(f"[InventoryFromForecast] p33={p33:.0f}  p67={p67:.0f}")

        # 4) Export standalone file
        out_path = self._build_output_path()
        self._exporter.export(out_path, rows)
        return out_path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_output_path(self) -> str:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        suffix = self._filename_suffix
        if suffix is None and not isinstance(self._adjuster, NoAdjustment):
            # Strip parentheses & dots so filename stays clean
            suffix = (
                self._adjuster.name
                .replace("(", "_")
                .replace(")", "")
                .replace(".", "")
            )
        parts = ["inventory_output", timestamp]
        if suffix:
            parts.append(suffix)
        return os.path.join(self._output_dir, "_".join(parts) + ".xlsx")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inventory_from_forecast.py <path_to_output.xlsx>")
        sys.exit(1)
    InventoryFromForecastPipeline(forecast_file=sys.argv[1]).run()