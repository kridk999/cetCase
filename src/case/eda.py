from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRICE_COLS = ["spot", "intraday1", "intraday2", "intraday3"]
SPREAD_COLS = ["spot_to_id1", "id1_to_id2", "id2_to_id3"]


def load_data(path: str) -> pd.DataFrame:
    """Loader data"""
    df = pd.read_csv(path)

    df['date_time'] = pd.to_datetime(df['date_time'],utc=True).dt.tz_convert('CET')
    df = df.set_index('date_time')

    df["Hour"] = df.index.tz_convert("Europe/Copenhagen").hour + 1
    
    return df


def add_features(df: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    """Tilføjer features som ratio, supply-demand gap, spreads og signal"""
    out = df.copy()

    out["ratio_wind_to_load"] = (
        out["wind_forecast_dah_mw"]
        / out["consumption_forecast_dah_mw"].replace(0, np.nan)
    )
    out["supply_demand_gap_mw"] = (
        out["wind_forecast_dah_mw"] - out["consumption_forecast_dah_mw"]
    )

    out["spot_to_id1"] = out["intraday1"] - out["spot"]
    out["id1_to_id2"] = out["intraday2"] - out["intraday1"]
    out["id2_to_id3"] = out["intraday3"] - out["intraday2"]

    out["signal_raw"] = 0
    out.loc[out["ratio_wind_to_load"] > q_high, "signal_raw"] = 1
    out.loc[out["ratio_wind_to_load"] < q_low, "signal_raw"] = -1

    out["cons_imbala"] = "neutral"
    out.loc[out["ratio_wind_to_load"] <= q_low, "cons_imbala"] = "scarcity_low_wind"
    out.loc[out["ratio_wind_to_load"] >= q_high, "cons_imbala"] = "surplus_high_wind"

    return out


def save_figure(fig: plt.Figure, path: Path, show: bool) -> None:
    """util til at gemme figur"""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_price_series(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    """Plotter en simpel tidsserie for spot og intraday priser"""
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in PRICE_COLS:
        ax.plot(df.index, df[col], label=col, linewidth=0.8)
    ax.set_title("Market Price Time Series")
    ax.set_ylabel("EUR/MWh")
    ax.legend()
    save_figure(fig, out_dir / "price_time_series.png", show)


def plot_ratio_distribution(df: pd.DataFrame, q_low: float, q_high: float, out_dir: Path, show: bool) -> None:
    """Plotter distributionen af ratio_wind men linjer for q_low og q_high"""
    vals = df["ratio_wind_to_load"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=80, alpha=0.85)
    ax.axvline(q_low, color="red", linestyle="--", linewidth=1.8, label=f"q_low={q_low}")
    ax.axvline(q_high, color="green", linestyle="--", linewidth=1.8, label=f"q_high={q_high}")
    ax.set_title("Distribution Of Wind/Load Ratio")
    ax.set_xlabel("ratio_wind_to_load")
    ax.set_ylabel("Count")
    ax.legend()
    save_figure(fig, out_dir / "ratio_distribution.png", show)


def plot_spread_box_by_regime(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    """Box plot af spreads opdelt på vind"""
    regimes = ["scarcity_low_wind", "neutral", "surplus_high_wind"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, leg in zip(axes, SPREAD_COLS):
        data = [df.loc[df["cons_imbala"] == r, leg].dropna() for r in regimes]
        ax.boxplot(data, tick_labels=regimes, showfliers=False)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{leg} by Consumption Imbalance")
        ax.set_ylabel("Spread")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2)

    save_figure(fig, out_dir / "spread_boxplots_by_regime.png", show)





def run_eda(input_path: str, output_dir: str, q_low: float, q_high: float, show: bool) -> None:
    """Kører hele EDA pipeline"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(input_path)
    df = add_features(df_raw, q_low=q_low, q_high=q_high)

    plot_price_series(df, out_dir, show)
    plot_ratio_distribution(df, q_low=q_low, q_high=q_high, out_dir=out_dir, show=show)
    plot_spread_box_by_regime(df, out_dir, show)



    print("EDA complete.")
    print(f"Output directory: {out_dir.resolve()}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extensive EDA for supply-demand mismatch and spread strategy support.")
    parser.add_argument("--input", default="assets/data.csv", help="Path to raw input CSV.")
    parser.add_argument("--output-dir", default="assets/eda", help="Directory to write plots/tables.")
    parser.add_argument("--q-low", type=float, default=0.12, help="Lower threshold for raw signal.")
    parser.add_argument("--q-high", type=float, default=0.85, help="Upper threshold for raw signal.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(
        input_path=args.input,
        output_dir=args.output_dir,
        q_low=args.q_low,
        q_high=args.q_high,
        show=args.show,
    )