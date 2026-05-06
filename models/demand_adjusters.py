"""
Demand Adjustment Strategies.

Applied between reading the forecast sheet and computing inventory metrics.
This is the "policy" layer that lets inventory planning diverge from raw
forecast values — without touching the forecast itself.

Strategy Pattern: implement IDemandAdjuster, plug into the pipeline.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class IDemandAdjuster(Protocol):
    """Contract: take forecast values, return adjusted values for inventory planning."""

    @property
    def name(self) -> str: ...

    def adjust(self, values: np.ndarray) -> np.ndarray: ...


class NoAdjustment:
    """Identity — use raw forecast values (default)."""
    name = "none"

    def adjust(self, values: np.ndarray) -> np.ndarray:
        return values


class MultiplicativeBuffer:
    """
    Apply a flat multiplier to all forecast values.
    Use for safety buffer (e.g., 1.10 = +10% planning buffer).
    """

    def __init__(self, factor: float):
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor}")
        self._factor = factor

    @property
    def name(self) -> str:
        return f"multiplicative({self._factor:.2f})"

    def adjust(self, values: np.ndarray) -> np.ndarray:
        return values * self._factor


class MonthlyMultiplicativeBuffer:
    """
    Per-position multipliers — different buffer per forecast month.
    Length must match forecast horizon.

    Example: holiday season buffer for months 10-12:
        MonthlyMultiplicativeBuffer([1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1.3, 1.25])
    """

    def __init__(self, factors: list[float]):
        if any(f <= 0 for f in factors):
            raise ValueError("all factors must be > 0")
        self._factors = np.array(factors, dtype=float)

    @property
    def name(self) -> str:
        return f"monthly({len(self._factors)} factors)"

    def adjust(self, values: np.ndarray) -> np.ndarray:
        if len(values) != len(self._factors):
            raise ValueError(
                f"factor length {len(self._factors)} != forecast length {len(values)}"
            )
        return values * self._factors