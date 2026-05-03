"""
Strategy Pattern: each forecasting model implements IForecastStrategy.
Adding a new model = create a new class, register in ForecastStrategyFactory.
"""
from __future__ import annotations

import abc
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from pmdarima import auto_arima
from prophet import Prophet


class IForecastStrategy(Protocol):
    """Contract every forecasting model must fulfil."""

    @property
    def name(self) -> str: ...

    def fit_predict_test(self, train: pd.Series, test: pd.Series) -> np.ndarray:
        """Train on `train`, return predictions aligned with `test`."""
        ...

    def forecast(self, ts: pd.Series, horizon: int) -> np.ndarray:
        """Train on full series `ts`, forecast `horizon` future periods."""
        ...


# ── Concrete strategies ────────────────────────────────────────────────────────

class ArimaStrategy:
    name = "ARIMA"

    def _build(self, series: pd.Series):
        return auto_arima(series, seasonal=True, m=12, stepwise=True, suppress_warnings=True)

    def fit_predict_test(self, train: pd.Series, test: pd.Series) -> np.ndarray:
        model = self._build(train)
        return np.array(model.predict(n_periods=len(test)))

    def forecast(self, ts: pd.Series, horizon: int) -> np.ndarray:
        model = self._build(ts)
        return np.array(model.predict(n_periods=horizon))


class ProphetStrategy:
    name = "Prophet"

    def _make_df(self, series: pd.Series) -> pd.DataFrame:
        return series.reset_index().rename(columns={"date": "ds", "demand": "y"})

    def _build_and_predict(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        model = Prophet(yearly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        fc = model.predict(future)
        return fc["yhat"].iloc[-horizon:].values

    def fit_predict_test(self, train: pd.Series, test: pd.Series) -> np.ndarray:
        df = self._make_df(train)
        return self._build_and_predict(df, len(test))

    def forecast(self, ts: pd.Series, horizon: int) -> np.ndarray:
        df = self._make_df(ts)
        return self._build_and_predict(df, horizon)


class RandomForestStrategy:
    name = "RandomForest"

    def _feature(self, series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame({"month": series.index.month})
        return X, series

    def fit_predict_test(self, train: pd.Series, test: pd.Series) -> np.ndarray:
        X_train, y_train = self._feature(train)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        X_test = pd.DataFrame({"month": test.index.month})
        return model.predict(X_test)

    def forecast(self, ts: pd.Series, horizon: int) -> np.ndarray:
        X, y = self._feature(ts)
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        future_months = pd.DataFrame({"month": [(i % 12) + 1 for i in range(horizon)]})
        return model.predict(future_months)


# ── Factory ────────────────────────────────────────────────────────────────────

class ForecastStrategyFactory:
    """
    Registry of available strategies.
    To add a new model: implement IForecastStrategy, add to _registry.
    """

    _registry: dict[str, IForecastStrategy] = {
        "ARIMA":        ArimaStrategy(),
        "Prophet":      ProphetStrategy(),
        "RandomForest": RandomForestStrategy(),
    }

    @classmethod
    def all_strategies(cls) -> list[IForecastStrategy]:
        return list(cls._registry.values())

    @classmethod
    def get(cls, name: str) -> IForecastStrategy:
        if name not in cls._registry:
            raise ValueError(f"Unknown strategy '{name}'. Available: {list(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def register(cls, strategy: IForecastStrategy) -> None:
        """Plug in a new strategy at runtime without changing existing code."""
        cls._registry[strategy.name] = strategy
