"""
ChartRenderer: all Matplotlib logic isolated in one module.
Adding a new chart = add one method, no changes to pipeline.
"""
from __future__ import annotations
from matplotlib.patches import Patch

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import COLORS


class ChartRenderer:
    def __init__(self, output_dir: str):
        self._dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = os.path.join(self._dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Chart 1: per-SKU forecast ─────────────────────────────────────────────

    def sku_forecast(
        self,
        sku_id: str,
        sku_name: str,
        train: pd.Series,
        test: pd.Series,
        test_pred: np.ndarray,
        fc_values: np.ndarray,
        fc_dates: list,
        model_used: str,
        best_mape: float,
        demand_level: str,
        variability: str,
    ) -> None:
        safe = str(sku_id).replace("/", "_")
        fc_std = float(np.std(fc_values))
        test_end = test.index.max()

        fig, ax = plt.subplots(figsize=(12, 5))
        self._plot_train(ax, train)
        self._plot_test_actual(ax, train, test)
        self._plot_forecast(ax, test, fc_values, fc_dates, fc_std, model_used)
        ax.axvline(test_end, color="#888", linewidth=1.2, linestyle=":", alpha=0.8)
        ax.text(test_end, ax.get_ylim()[1], " Forecast start", fontsize=8, color="#888", va="top")
        ax.set_title(
            f"{sku_name}  [SKU: {sku_id}]"
            f"   |   Demand: {demand_level}   •   Variability: {variability}"
            f"   •   Best Model: {model_used}"
        )
        self._format_ax(ax)
        self._save(fig, f"forecast_{safe}_notest.png")

        # Re-open same figure to overlay train/test split info
        fig, ax = plt.subplots(figsize=(12, 5))
        self._plot_train(ax, train)
        self._plot_test_actual(ax, train, test)
        self._plot_test_pred(ax, train, test, test_pred, model_used, best_mape)
        self._plot_forecast(ax, test, fc_values, fc_dates, fc_std, model_used)
        ax.axvline(train.index.max(), color="gray", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axvline(test_end, color="#888", linewidth=1.2, linestyle=":", alpha=0.8)
        ax.set_title(
            f"{sku_name}  [SKU: {sku_id}]"
            f"   |   Demand: {demand_level}   •   Variability: {variability}"
            f"   •   Best Model: {model_used}"
        )
        self._format_ax(ax)
        self._save(fig, f"forecast_{safe}.png")

    # ── Chart 2: model distribution ───────────────────────────────────────────

    def model_distribution(self, eval_rows: list[dict]) -> None:
        import pandas as pd
        df = pd.DataFrame(eval_rows)
        model_counts = df["best_model"].value_counts()
        all_models = ["ARIMA", "Prophet", "RandomForest"]
        counts = [model_counts.get(m, 0) for m in all_models]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        bars = axes[0].bar(all_models, counts, color=[COLORS[m] for m in all_models],
                           width=0.5, edgecolor="white")
        axes[0].bar_label(bars, fmt="%d", fontsize=11, fontweight="bold", padding=3)
        axes[0].set_title("Best Model Count per SKU")
        axes[0].set_ylabel("Number of SKUs")
        axes[0].set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 5)

        mape_map = {"ARIMA": "ARIMA_MAPE", "Prophet": "Prophet_MAPE", "RandomForest": "RF_MAPE"}
        box_data, box_labels = [], []
        for m in all_models:
            col = mape_map[m]
            vals = df[col].replace(np.inf, np.nan).dropna().values
            if len(vals):
                box_data.append(vals)
                box_labels.append(m)

        bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True,
                             medianprops=dict(color="white", linewidth=2))
        for patch, m in zip(bp["boxes"], box_labels):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.75)
        axes[1].set_title("MAPE Distribution by Model")
        axes[1].set_ylabel("MAPE")
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

        fig.suptitle("Model Selection Summary", fontsize=13, fontweight="bold", y=1.01)
        self._save(fig, "model_distribution.png")

    # ── Chart 3: SKU classification ───────────────────────────────────────────

    def sku_classification(self, inv_df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        dl = inv_df["demand_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        bars = axes[0].bar(dl.index, dl.values, color=[COLORS[k.lower()] for k in dl.index],
                           width=0.5, edgecolor="white")
        axes[0].bar_label(bars, fmt="%d", fontsize=11, fontweight="bold", padding=3)
        axes[0].set_title("SKU Demand Level")
        axes[0].set_ylabel("Number of SKUs")
        axes[0].set_ylim(0, dl.max() * 1.3 if dl.max() > 0 else 5)

        vr = inv_df["variability"].value_counts().reindex(["Stable", "Volatile"], fill_value=0)
        bars = axes[1].bar(vr.index, vr.values, color=[COLORS[k.lower()] for k in vr.index],
                           width=0.4, edgecolor="white")
        axes[1].bar_label(bars, fmt="%d", fontsize=11, fontweight="bold", padding=3)
        axes[1].set_title("SKU Demand Variability")
        axes[1].set_ylabel("Number of SKUs")
        axes[1].set_ylim(0, vr.max() * 1.3 if vr.max() > 0 else 5)

        lc = {"High": COLORS["high"], "Medium": COLORS["medium"], "Low": COLORS["low"]}
        for level, grp in inv_df.groupby("demand_level"):
            axes[2].scatter(grp["annual_demand"], grp["cv"], c=lc[level], label=level,
                            s=70, edgecolors="white", linewidths=0.5, zorder=3)
        axes[2].axhline(0.3, color=COLORS["volatile"], linewidth=1.2,
                        linestyle="--", alpha=0.7, label="CV=0.3 threshold")
        axes[2].set_title("Annual Demand vs CV (Variability)")
        axes[2].set_xlabel("Annual Demand")
        axes[2].set_ylabel("CV (std / mean)")
        axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        axes[2].legend(fontsize=8)
        for _, row in inv_df.iterrows():
            axes[2].annotate(str(row["sku_id"]), (row["annual_demand"], row["cv"]),
                             fontsize=7, textcoords="offset points", xytext=(4, 3), color="gray")

        fig.suptitle("SKU Classification Overview", fontsize=13, fontweight="bold", y=1.01)
        self._save(fig, "sku_classification.png")

    # ── Chart 4: inventory parameters ────────────────────────────────────────

    def inventory_parameters(self, inv_df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        x = range(len(inv_df))
        x_ticks = [str(v) for v in inv_df["sku_id"]]
        for ax, col, title, color in [
            (axes[0], "eoq",          "EOQ per SKU",           COLORS["ARIMA"]),
            (axes[1], "safety_stock", "Safety Stock per SKU",  COLORS["Prophet"]),
            (axes[2], "rop",          "Reorder Point per SKU", COLORS["RandomForest"]),
        ]:
            bars = ax.bar(x, inv_df[col], color=color, edgecolor="white", width=0.6)
            ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
            ax.set_title(title)
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_ticks, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Units")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig.suptitle("Inventory Policy Parameters", fontsize=13, fontweight="bold", y=1.01)
        self._save(fig, "inventory_parameters.png")

    # ── Chart 5: all-SKU summary ──────────────────────────────────────────────

    def all_skus_summary(
        self,
        sku_data: list[dict],   # list of dicts with all per-SKU data
        inv_df: pd.DataFrame,
    ) -> None:
        n = len(sku_data)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

        for idx, d in enumerate(sku_data):
            ax = axes[idx // ncols][idx % ncols]
            sku_id = d["sku_id"]
            row = inv_df[inv_df["sku_id"] == sku_id].iloc[0]

            self._plot_train(ax, d["train"])
            self._plot_test_actual(ax, d["train"], d["test"])
            if len(d["test_pred"]) > 0:
                self._plot_test_pred(ax, d["train"], d["test"], d["test_pred"],
                                     d["model_used"], d["best_mape"], fontsize=7)
            self._plot_forecast(ax, d["test"], d["fc_values"], d["fc_dates"],
                                float(np.std(d["fc_values"])), d["model_used"])
            ax.axvline(d["train"].index.max(), color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.axvline(d["test"].index.max(), color="#888", linewidth=0.8, linestyle=":", alpha=0.7)
            ax.set_title(
                f"{sku_id} | {d['model_used']} | {row['demand_level']}/{row['variability']}",
                fontsize=9,
            )
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.margins(x=0.01)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("Forecast Summary — All SKUs (Detailed)", fontsize=13, fontweight="bold")
        self._save(fig, "forecast_all_skus.png")

        # ── Chart 6: Total historical demand across all SKUs ─────────────────────
 
    def total_historical_demand(self, chart_data: list[dict]) -> None:
        """Stacked area + total line — all SKUs combined over history."""
        all_series = {}
        for d in chart_data:
            ts = pd.concat([d["train"], d["test"]])
            all_series[d["sku_id"]] = ts
 
        combined = pd.DataFrame(all_series).sort_index()
        total = combined.sum(axis=1)
 
        fig, ax = plt.subplots(figsize=(13, 5))
        sku_colors = list(COLORS[k] for k in ["ARIMA", "Prophet", "RandomForest", "band"])
        bottom = np.zeros(len(combined))
        for i, sku_id in enumerate(combined.columns):
            vals = combined[sku_id].values
            color = sku_colors[i % len(sku_colors)]
            ax.fill_between(combined.index, bottom, bottom + vals,
                            alpha=0.55, color=color, label=str(sku_id))
            bottom += vals
 
        ax.plot(total.index, total.values,
                color="#2d2d2d", linewidth=2, linestyle="-",
                label="Total", zorder=5)
        ax.scatter(total.index, total.values,
                   color="#2d2d2d", s=18, zorder=6)
 
        ax.set_title("Total Historical Demand — All SKUs")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand (units)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(loc="upper left", fontsize=8.5)
        ax.margins(x=0.01)
        self._save(fig, "total_historical_demand.png")
 
    # ── Chart 7: Total forecast demand across all SKUs ────────────────────────
 
    def total_forecast_demand(self, chart_data: list[dict]) -> None:
        """Stacked area of forecast per SKU + total line."""
        # Align all forecast series on the same date index
        fc_series: dict[str, pd.Series] = {}
        for d in chart_data:
            fc_series[d["sku_id"]] = pd.Series(
                d["fc_values"], index=pd.DatetimeIndex(d["fc_dates"])
            )
 
        combined_fc = pd.DataFrame(fc_series).sort_index()
        total_fc = combined_fc.sum(axis=1)
 
        # Last actual point of each SKU for connecting line
        last_actuals: dict[str, float] = {}
        last_dates: dict[str, pd.Timestamp] = {}
        for d in chart_data:
            ts_full = pd.concat([d["train"], d["test"]])
            last_dates[d["sku_id"]] = ts_full.index.max()
            last_actuals[d["sku_id"]] = float(ts_full.values[-1])
 
        sku_colors = list(COLORS[k] for k in ["ARIMA", "Prophet", "RandomForest", "band"])
        fig, ax = plt.subplots(figsize=(13, 5))
 
        bottom = np.zeros(len(combined_fc))
        for i, sku_id in enumerate(combined_fc.columns):
            vals = combined_fc[sku_id].values
            color = sku_colors[i % len(sku_colors)]
            ax.fill_between(combined_fc.index, bottom, bottom + vals,
                            alpha=0.55, color=color, label=str(sku_id))
            bottom += vals
 
        ax.plot(total_fc.index, total_fc.values,
                color="#2d2d2d", linewidth=2, linestyle="--",
                label="Total forecast", zorder=5)
        ax.scatter(total_fc.index, total_fc.values,
                   color="#2d2d2d", s=18, zorder=6)
 
        # Uncertainty band on total
        total_arr = total_fc.values
        fc_std = float(np.std(total_arr))
        ax.fill_between(combined_fc.index,
                        np.maximum(total_arr - fc_std, 0),
                        total_arr + fc_std,
                        color=COLORS["band"], alpha=0.15,
                        label=f"±1 std band (σ={fc_std:,.0f})")
 
        ax.set_title("Total Forecast Demand — All SKUs (Next 12 Months)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand (units)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(loc="upper left", fontsize=8.5)
        ax.margins(x=0.01)
        self._save(fig, "total_forecast_demand.png")
 
    # ── Chart 8: Average MAPE per model (bar + value labels) ─────────────────

    def avg_mape_by_model(self, eval_rows: list[dict]) -> None:
        """Grouped bar: avg MAPE of each model across all SKUs, with individual SKU dots."""
        df = pd.DataFrame(eval_rows)
        all_models = ["ARIMA", "Prophet", "RandomForest"]
        mape_cols = {"ARIMA": "ARIMA_MAPE", "Prophet": "Prophet_MAPE", "RandomForest": "RF_MAPE"}
 
        avg_mapes = {
            m: df[mape_cols[m]].replace(np.inf, np.nan).mean()
            for m in all_models
        }
 
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(all_models))
        bar_w = 0.5
        bars = ax.bar(x, [avg_mapes[m] for m in all_models],
                      width=bar_w, color=[COLORS[m] for m in all_models],
                      edgecolor="white", zorder=2)
        ax.bar_label(bars,
                     labels=[f"{avg_mapes[m]:.1%}" for m in all_models],
                     fontsize=11, fontweight="bold", padding=4)
 
        # Scatter individual SKU MAPE values over bars
        for i, m in enumerate(all_models):
            col = mape_cols[m]
            vals = df[col].replace(np.inf, np.nan).dropna().values
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(i + jitter, vals,
                       color="white", edgecolors=COLORS[m],
                       s=50, linewidth=1.5, zorder=4, label=None)
 
        ax.set_xticks(x)
        ax.set_xticklabels(all_models, fontsize=11)
        ax.set_title("Average MAPE by Model (across all SKUs)")
        ax.set_ylabel("MAPE")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.set_ylim(0, max(avg_mapes.values()) * 1.4)
        ax.margins(x=0.2)
        self._save(fig, "avg_mape_by_model.png")
 
    # ── Chart 9: Per-SKU MAPE comparison (all models) ────────────────────────
 
    def sku_mape_comparison(self, sku_id: str, mape_scores: dict[str, float]) -> None:
        """Horizontal bar showing MAPE of each model for one SKU, best model highlighted."""
        safe = str(sku_id).replace("/", "_")
        all_models = ["ARIMA", "Prophet", "RandomForest"]
        best = min(mape_scores, key=mape_scores.get)
 
        labels, values, colors = [], [], []
        for m in all_models:
            score = mape_scores.get(m, np.inf)
            if not np.isinf(score):
                labels.append(m)
                values.append(score)
                alpha_color = COLORS[m] if m == best else COLORS[m] + "80"
                colors.append(COLORS[m])
 
        fig, ax = plt.subplots(figsize=(8, 3.5))
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors, edgecolor="white", height=0.5)
 
        for bar, label, val, m in zip(bars, labels, values, [l for l in labels]):
            alpha = 1.0 if m == best else 0.45
            bar.set_alpha(alpha)
            ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=9, fontweight="bold",
                    color="#333333")
 
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.set_xlabel("MAPE")
        ax.set_title(f"[{sku_id}] Model MAPE Comparison   •   Best: {best}")
        ax.invert_yaxis()
 
        # Star annotation on best
        best_idx = labels.index(best)
        ax.annotate("★ best", xy=(values[best_idx], best_idx),
                    xytext=(values[best_idx] + max(values) * 0.15, best_idx),
                    fontsize=8, color=COLORS[best], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS[best], lw=1))
 
        fig.tight_layout()
        self._save(fig, f"mape_comparison_{safe}.png")
 
    # ── Chart 10: Per-SKU historical demand (standalone) ─────────────────────
 
    def sku_historical_demand(
        self,
        sku_id: str,
        sku_name: str,
        train: pd.Series,
        test: pd.Series,
    ) -> None:
        """Clean standalone historical demand chart — train + test, no forecast overlay."""
        safe = str(sku_id).replace("/", "_")
        ts_full = pd.concat([train, test]).sort_index()
 
        fig, ax = plt.subplots(figsize=(12, 4.5))
 
        # Train region shading
        ax.axvspan(train.index.min(), train.index.max(),
                   color=COLORS["history"], alpha=0.06, label="Train period")
        # Test region shading
        ax.axvspan(train.index.max(), test.index.max(),
                   color=COLORS["test_pred"], alpha=0.06, label="Test period")
 
        ax.plot(ts_full.index, ts_full.values,
                color=COLORS["history"], linewidth=1.8, zorder=3)
        ax.scatter(ts_full.index, ts_full.values,
                   color=COLORS["history"], s=22, zorder=4)
 
        # 3-month rolling average
        rolling = ts_full.rolling(3, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values,
                color=COLORS["forecast"], linewidth=1.5, linestyle="--",
                alpha=0.8, label="3-month rolling avg", zorder=3)
 
        # Train / test divider
        ax.axvline(train.index.max(), color="gray", linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.text(train.index.max(), ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else ts_full.max() * 1.05,
                " Train / Test", fontsize=8, color="gray", va="top")
 
        ax.set_title(f"{sku_name}  [SKU: {sku_id}]  — Historical Demand")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand (units)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(loc="upper left", fontsize=8.5)
        ax.margins(x=0.01)
        self._save(fig, f"historical_demand_{safe}.png")

    # ── Chart 11: Total cost by SKU  ─────────────────────
    def total_cost_by_sku(self, inv_df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(inv_df))
        colors = [COLORS[lvl.lower()] for lvl in inv_df["demand_level"]]
        bars = ax.bar(x, inv_df["total_cost"], color=colors, edgecolor="white", width=0.5)
        ax.bar_label(bars,
                    labels=[f"${v:,.0f}" for v in inv_df["total_cost"]],
                    fontsize=9, fontweight="bold", padding=4)
        ax.set_xticks(x)
        ax.set_xticklabels(inv_df["sku_id"], fontsize=11)
        ax.set_title("Total Inventory Cost by SKU")
        ax.set_ylabel("Total Cost (USD/year)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

        legend_patches = [
            Patch(color=COLORS["high"],   label="High demand"),
            Patch(color=COLORS["medium"], label="Medium demand"),
            Patch(color=COLORS["low"],    label="Low demand"),
        ]
        ax.legend(handles=legend_patches, fontsize=8.5)
        ax.margins(x=0.15)
        self._save(fig, "total_cost_by_sku.png")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _plot_train(self, ax, train: pd.Series) -> None:
        ax.plot(train.index, train.values, color=COLORS["history"], linewidth=1.8,
                label="Historical sales (train)", zorder=3)
        ax.scatter(train.index, train.values, color=COLORS["history"], s=22, zorder=4)

    def _plot_test_actual(self, ax, train: pd.Series, test: pd.Series) -> None:
        conn_idx = [train.index[-1]] + list(test.index)
        conn_vals = [train.values[-1]] + list(test.values)
        ax.plot(conn_idx, conn_vals, color=COLORS["history"], linewidth=1.8, zorder=3)
        ax.scatter(test.index, test.values, color=COLORS["history"], s=28, zorder=4)

    def _plot_test_pred(
        self, ax, train: pd.Series, test: pd.Series,
        test_pred: np.ndarray, model_used: str, best_mape: float, fontsize: int = 9,
    ) -> None:
        conn_idx = [train.index[-1]] + list(test.index)
        conn_vals = [train.values[-1]] + list(test_pred)
        ax.plot(conn_idx, conn_vals, color=COLORS["test_pred"], linewidth=2,
                linestyle="-.", label=f"{model_used} test prediction (MAPE={best_mape:.1%})", zorder=3)
        ax.scatter(test.index, test_pred, color=COLORS["test_pred"], s=28, marker="D", zorder=4)
        ax.fill_between(test.index, test.values, test_pred,
                        color=COLORS["test_pred"], alpha=0.12, label="Model error (test)")
        mid = test.index[len(test) // 2]
        mid_y = (test.values.mean() + test_pred.mean()) / 2
        ax.annotate(f"MAPE: {best_mape:.1%}", xy=(mid, mid_y), fontsize=fontsize,
                    color=COLORS["test_pred"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=COLORS["test_pred"], alpha=0.8))

    def _plot_forecast(
        self, ax, test: pd.Series, fc_values: np.ndarray,
        fc_dates: list, fc_std: float, model_used: str,
    ) -> None:
        conn_idx = [test.index.max()] + list(fc_dates)
        conn_vals = [test.values[-1]] + list(fc_values)
        ax.plot(conn_idx, conn_vals, color=COLORS["forecast"], linewidth=2, linestyle="--",
                label=f"Forecast ({model_used})", zorder=3)
        ax.scatter(fc_dates, fc_values, color=COLORS["forecast"], s=22, zorder=4)
        ax.fill_between(fc_dates, np.maximum(fc_values - fc_std, 0), fc_values + fc_std,
                        color=COLORS["band"], alpha=0.18, label=f"±1 std band (σ={fc_std:.1f})")

    def _format_ax(self, ax) -> None:
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand (units)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(loc="upper left", fontsize=8.5)
        ax.margins(x=0.01)
