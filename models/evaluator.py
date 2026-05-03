"""
ModelEvaluator: runs all strategies against the test window,
picks the winner by MAPE, then re-trains on full series to forecast.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from core.models import ForecastResult, ModelEvalResult
from models.forecast_strategies import ForecastStrategyFactory


class ModelEvaluator:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.strategies = ForecastStrategyFactory.all_strategies()

    def evaluate(
        self,
        sku_id: str,
        sku_name: str,
        train: pd.Series,
        test: pd.Series,
        ts: pd.Series,
    ) -> tuple[ModelEvalResult, ForecastResult]:

        mape_scores: dict[str, float] = {}
        predictions: dict[str, np.ndarray] = {}

        for strategy in self.strategies:
            try:
                preds = np.maximum(strategy.fit_predict_test(train, test), 0)
                mape = mean_absolute_percentage_error(test.values, preds)
            except Exception as exc:
                print(f"  [{strategy.name} error SKU {sku_id}]: {exc}")
                preds = np.array([])
                mape = np.inf

            mape_scores[strategy.name] = round(mape, 4)
            predictions[strategy.name] = preds

        best_name = min(mape_scores, key=mape_scores.get)
        best_strategy = ForecastStrategyFactory.get(best_name)

        # Forecast future using best strategy on full series
        future_values = np.maximum(best_strategy.forecast(ts, self.horizon), 0)
        last_date = ts.index.max()
        future_dates = [
            last_date + pd.DateOffset(months=i + 1) for i in range(self.horizon)
        ]

        eval_result = ModelEvalResult(
            sku_id=sku_id,
            sku_name=sku_name,
            mape_scores=mape_scores,
            best_model=best_name,
            test_predictions=predictions[best_name],
        )

        forecast_result = ForecastResult(
            sku_id=sku_id,
            sku_name=sku_name,
            dates=future_dates,
            values=future_values,
            model_used=best_name,
        )

        return eval_result, forecast_result