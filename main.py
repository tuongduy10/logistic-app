"""
Entry point — keep this file minimal.

Stage 1: LogisticPipeline             → exports output_<ts>.xlsx
                                        (forecast + eval + inventory at face value)

Stage 2: InventoryFromForecastPipeline → reads forecast_output sheet,
                                         applies business policy (adjustment + params),
                                         writes inventory_output_<ts>_<policy>.xlsx
"""
import warnings
warnings.filterwarnings("ignore")

from pipeline import LogisticPipeline
from models.inventory_from_forecast import InventoryFromForecastPipeline
from models.demand_adjusters import MultiplicativeBuffer


if __name__ == "__main__":
    # ── Stage 1: forecast + face-value inventory ──────────────────────────────
    forecast_output_path = LogisticPipeline().run()

    # ── Stage 2: inventory under planning policy ──────────────────────────────
    print("\n" + "=" * 70)
    print("Stage 2: Recomputing inventory from forecast_output sheet")
    print("=" * 70)

    # Default planning buffer: +10% on top of raw forecast
    InventoryFromForecastPipeline(
        forecast_file=forecast_output_path,
        demand_adjuster=MultiplicativeBuffer(factor=1.10),
        # inventory_params_file="adjusted_params_q4.xlsx",  # optional override
    ).run()