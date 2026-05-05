"""
Entry point — keep this file minimal.

Stage 1: LogisticPipeline             → exports output_<ts>.xlsx (forecast + eval + inventory)
Stage 2: InventoryFromForecastPipeline → reads forecast_output sheet from Stage 1's file,
                                         recomputes inventory, writes inventory_output_<ts>.xlsx
"""
import warnings
warnings.filterwarnings("ignore")

from pipeline import LogisticPipeline
from models.inventory_from_forecast import InventoryFromForecastPipeline


if __name__ == "__main__":
    # Stage 1 — full forecast + inventory pipeline (existing behaviour)
    forecast_output_path = LogisticPipeline().run()

    # Stage 2 — independent inventory file derived from forecast_output sheet
    print("\n" + "=" * 70)
    print("Stage 2: Recomputing inventory from forecast_output sheet")
    print("=" * 70)
    InventoryFromForecastPipeline(forecast_file=forecast_output_path).run()