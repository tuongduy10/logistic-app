"""
Pipeline: orchestrates the full workflow.
Each stage is clearly separated — easy to add/remove steps.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    COLORS, FORECAST_HORIZON, INPUT_FILE,
    MPL_THEME, OUTPUT_CHARTS_DIR, OUTPUT_FILES_DIR,
)
from charts.renderer import ChartRenderer
from core.repository import (
    ExcelExporter, InventoryParamsRepository, SalesRepository,
)
from models.evaluator import ModelEvaluator
from models.inventory import (
    InventoryCalculator, demand_level_label,
)


class LogisticPipeline:
    """
    Orchestrates: Load → Model Eval → Forecast → Inventory → Charts → Export.
    Each stage delegates to a dedicated component — no business logic here.
    """

    def __init__(self):
        plt.rcParams.update(MPL_THEME)
        os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)

        self._sales_repo = SalesRepository(INPUT_FILE)
        self._inv_repo = InventoryParamsRepository(INPUT_FILE)
        self._evaluator = ModelEvaluator(horizon=FORECAST_HORIZON)
        self._inv_calc = InventoryCalculator()
        self._renderer = ChartRenderer(OUTPUT_CHARTS_DIR)
        self._exporter = ExcelExporter()

    def run(self) -> str:
        """Run the full pipeline and return the path of the exported Excel file."""
        sku_ids = self._sales_repo.sku_ids()
        print(f"[Pipeline] Processing {len(sku_ids)} SKUs...\n")

        eval_rows: list[dict] = []
        forecast_rows: list[dict] = []
        inventory_rows: list[dict] = []
        chart_data: list[dict] = []   # per-SKU bundles for visualisation

        # ── Stage 1: Per-SKU model evaluation + forecast + inventory ──────────
        for sku_id in sku_ids:
            ts, train, test, sku_name = self._sales_repo.get_sku_series(sku_id)
            inv_params = self._inv_repo.get(sku_id)

            eval_result, forecast = self._evaluator.evaluate(
                sku_id, sku_name, train, test, ts
            )
            inv_result = self._inv_calc.compute(forecast, inv_params)
            best_mape = eval_result.mape_scores[eval_result.best_model]

            # Collect rows for Excel export
            eval_rows.append({
                "sku_id":       sku_id,
                "sku_name":     sku_name,
                "ARIMA_MAPE":   eval_result.mape_scores.get("ARIMA", np.inf),
                "Prophet_MAPE": eval_result.mape_scores.get("Prophet", np.inf),
                "RF_MAPE":      eval_result.mape_scores.get("RandomForest", np.inf),
                "best_model":   eval_result.best_model,
            })
            for i, val in enumerate(forecast.values):
                forecast_rows.append({
                    "date":            forecast.dates[i],
                    "sku_id":          sku_id,
                    "sku_name":        sku_name,
                    "forecast_demand": round(val, 2),
                    "model_used":      forecast.model_used,
                })
            inventory_rows.append(vars(inv_result))

            # Store chart data
            chart_data.append({
                "sku_id":    sku_id,
                "sku_name":  sku_name,
                "train":     train,
                "test":      test,
                "test_pred": eval_result.test_predictions,
                "fc_values": forecast.values,
                "fc_dates":  forecast.dates,
                "model_used": eval_result.best_model,
                "best_mape": best_mape,
            })

            print(
                f"  ✓ {sku_id} | {sku_name} | best={eval_result.best_model}"
                f" | MAPE={best_mape:.3f} | variability={inv_result.variability}"
            )

        # ── Stage 2: Cross-SKU demand level classification ────────────────────
        inv_df = pd.DataFrame(inventory_rows)
        annual = inv_df["annual_demand"].values
        p33, p67 = float(np.percentile(annual, 33)), float(np.percentile(annual, 67))
        inv_df["demand_level"] = inv_df["annual_demand"].apply(
            lambda d: demand_level_label(d, p33, p67)
        )
        # Push demand_level back into inventory_rows for export
        dl_map = inv_df.set_index("sku_id")["demand_level"].to_dict()
        for row in inventory_rows:
            row["demand_level"] = dl_map.get(row["sku_id"], "")

        print(f"\n[Classification] p33={p33:.0f}  p67={p67:.0f}")
        print(
            inv_df[["sku_id", "sku_name", "annual_demand", "demand_level",
                     "variability", "cv"]].to_string(index=False)
        )

        # ── Stage 3: Visualisation ─────────────────────────────────────────────
        print("\n[Charts] Rendering...")

        for d in chart_data:
            sku_id = d["sku_id"]
            inv_row = inv_df[inv_df["sku_id"] == sku_id].iloc[0]
            best_mape_val = d["best_mape"]
            mape_scores = {
                "ARIMA":        next(r["ARIMA_MAPE"]   for r in eval_rows if r["sku_id"] == sku_id),
                "Prophet":      next(r["Prophet_MAPE"] for r in eval_rows if r["sku_id"] == sku_id),
                "RandomForest": next(r["RF_MAPE"]      for r in eval_rows if r["sku_id"] == sku_id),
            }

            # Existing charts
            self._renderer.sku_forecast(
                sku_id=sku_id,
                sku_name=d["sku_name"],
                train=d["train"],
                test=d["test"],
                test_pred=d["test_pred"],
                fc_values=d["fc_values"],
                fc_dates=d["fc_dates"],
                model_used=d["model_used"],
                best_mape=best_mape_val,
                demand_level=inv_row["demand_level"],
                variability=inv_row["variability"],
            )

            # New per-SKU charts
            self._renderer.sku_mape_comparison(sku_id, mape_scores)
            self._renderer.sku_historical_demand(
                sku_id=sku_id,
                sku_name=d["sku_name"],
                train=d["train"],
                test=d["test"],
            )

        # Existing global charts
        self._renderer.model_distribution(eval_rows)
        self._renderer.sku_classification(inv_df)
        self._renderer.inventory_parameters(inv_df)
        self._renderer.all_skus_summary(chart_data, inv_df)

        # New global charts
        self._renderer.total_historical_demand(chart_data)
        self._renderer.total_forecast_demand(chart_data)
        self._renderer.avg_mape_by_model(eval_rows)
        self._renderer.total_cost_by_sku(inv_df)

        # ── Stage 4: Export ────────────────────────────────────────────────────
        output_path = os.path.join(
            OUTPUT_FILES_DIR,
            f"output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        )
        self._exporter.export(output_path, forecast_rows, eval_rows, inventory_rows)

        print(f"\n✓ Done!  Charts → ./{OUTPUT_CHARTS_DIR}/")
        return output_path