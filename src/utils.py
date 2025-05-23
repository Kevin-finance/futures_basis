from datetime import datetime, date
from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from functools import reduce
from plotly.subplots import make_subplots


def merge_and_filter(
    dfs: List[pl.DataFrame],
    on: str,
    strategy: str,
    tolerance: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    def _join(left, right):
        return left.join_asof(right, on=on, strategy=strategy, tolerance=tolerance)

    merged = reduce(_join, dfs)
    return merged.filter((pl.col(on) >= start) & (pl.col(on) <= end))


def compute_adjustments(df: pl.DataFrame) -> pl.DataFrame:
    return df


def compute_theoretical_prices(df: pl.DataFrame) -> pl.DataFrame:
    # Formula:
    # Theoretical Futures Price = S × exp[(r + s) × T] − D
    # where:
    #   S = Spot index price ("Close")
    #   r = Near-term interest rate (in decimal, i.e., "Near_Rate" / 100)
    #   s = Financing spread (in decimal, i.e., "Financing_Spread" / 10000)
    #   T = Time to maturity in years ("NearMonth_Years")
    #   D = Expected discounted dividends ("Expected Discounted Points")
    
    return df.with_columns(
        (
            pl.col("Close")
            * (
                np.exp(
                    ((pl.col("Near_Rate") / 100) + (pl.col("Financing_Spread") / 10000))
                    * pl.col("NearMonth_Years")
                )
            )
            - (pl.col("Expected Discounted Points"))
        ).alias("Theoretical Futures Price")
    )


def finalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.rename(
        {
            "Close": "Spot",
            "Expected Points": "Expected Dividend Points",
            "Last": "ES Trade",
            "Last_right": "BTIC",
        }
    )
    df = df.with_columns(
        [
            (pl.col("Theoretical Futures Price") - pl.col("Spot")).alias(
                "Theoretical Basis"
            ),
            (pl.col("ES Trade") - pl.col("Spot")).alias("Market Basis"),
        ]
    )

    return df.with_columns(
        [
            ((pl.col("Theoretical Basis") * 10000) / pl.col("Spot")).alias(
                "Theoretical Basis Rate"
            ),
            ((pl.col("Market Basis") * 10000) / pl.col("Spot")).alias(
                "Market Basis Rate"
            ),
            ((pl.col("BTIC") * 10000) / pl.col("Spot")).alias("BTIC Rate"),
        ]
    )


def plot_basis(df_pd: pd.DataFrame, in_bps: bool = False):
    output_pd = df_pd
    output_pd["UTC-Datetime"] = pd.to_datetime(output_pd["UTC-Datetime"])
    color_dict = {
        "BTIC": dict(color="rgba(100, 200, 100, 0.3)"),
        "Theoretical Basis": dict(color="rgba(0, 100, 200, 0.3)"),
        "Market Basis": dict(color="rgba(100, 100, 200, 0.3)"),
    }
    fig = go.Figure()
    for k, color in color_dict.items():
        col = k + " Rate" if in_bps else k
        fig.add_trace(
            go.Scatter(
                x=output_pd["UTC-Datetime"],
                y=output_pd[col],
                mode="lines+markers",
                name=col,
                line=color,
            )
        )
    title = "BTIC vs Theoretical Basis vs Market Basis"
    if in_bps:
        title = title + " (Rates(bp))"
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Basis Rate" if in_bps else "Index Points",
        legend_title="Legend",
        hovermode="x unified",
        template="plotly_white",
        height=600,
    )

    return fig


def plot_basis_dual(df_pd: pd.DataFrame, in_bps: bool = False):
    output_pd = df_pd.copy()
    output_pd["UTC-Datetime"] = pd.to_datetime(output_pd["UTC-Datetime"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if in_bps:
        fig.add_trace(
            go.Scatter(
                x=output_pd["UTC-Datetime"],
                y=output_pd["Theoretical Basis Rate"],
                mode="lines+markers",
                name="Theoretical Basis Rate",
                line=dict(color="rgba(0, 100, 200, 0.8)"),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=output_pd["UTC-Datetime"],
                y=output_pd["Market Basis Rate"],
                mode="lines+markers",
                name="Market Basis Rate",
                line=dict(color="rgba(100, 100, 200, 0.5)"),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=output_pd["UTC-Datetime"],
                y=output_pd["BTIC Rate"],
                mode="lines+markers",
                name="BTIC Rate",
                line=dict(color="rgba(100, 200, 100, 0.5)"),
            ),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Theoretical Basis Rate (bps)", secondary_y=False)
        fig.update_yaxes(title_text="Market & BTIC Basis Rate (bps)", secondary_y=True)
        fig.update_layout(title="Basis Rate Comparison (bps)")

    else:
        fig.add_trace(
            go.Scatter(
                x=output_pd["UTC-Datetime"],
                y=output_pd["Theoretical Basis"],
                mode="lines+markers",
                name="Theoretical Basis",
                line=dict(color="rgba(0, 100, 200, 0.8)"),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=output_pd["UTC-Datetime"],
                y=output_pd["Market Basis"],
                mode="lines+markers",
                name="Market Basis",
                line=dict(color="rgba(100, 100, 200, 0.5)"),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=output_pd["UTC-Datetime"],
                y=output_pd["BTIC"],
                mode="lines+markers",
                name="BTIC",
                line=dict(color="rgba(100, 200, 100, 0.5)"),
            ),
            secondary_y=True,
        )

        fig.update_yaxes(
            title_text="Theoretical Basis (Index Points)", secondary_y=False
        )
        fig.update_yaxes(title_text="Market & BTIC (Index Points)", secondary_y=True)
        fig.update_layout(title="Basis Comparison (Index Points)")

    fig.update_layout(
        xaxis_title="Time",
        legend_title="Legend",
        hovermode="x unified",
        template="plotly_white",
        height=600,
    )

    return fig


def plot_spreads(df):
    ts = df["UTC-Datetime"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=df["Market Basis"].to_numpy(), name="ES - Spot"))
    fig.add_trace(
        go.Scatter(x=ts, y=df["Theoretical Basis"].to_numpy(), name="Theo - Spot")
    )
    fig.add_trace(
        go.Scatter(x=ts, y=df["BTIC"].to_numpy(), name="BTIC", line=dict(dash="dot"))
    )
    fig.update_layout(
        title="Spread to Spot", template="plotly_white", hovermode="x unified"
    )

    return fig


def plot_percentage_changes(df: pl.DataFrame):
    es0, theo0, btic0 = df["Market Basis"][0], df["Theoretical Basis"][0], df["BTIC"][0]
    ts = df["UTC-Datetime"].to_numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ts, y=((df["Market Basis"] - es0) / es0 * 100), name="ES % Δ")
    )
    fig.add_trace(
        go.Scatter(
            x=ts, y=((df["Theoretical Basis"] - theo0) / theo0 * 100), name="Theo % Δ"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=((df["BTIC"] - btic0) / btic0 * 100),
            name="BTIC % Δ",
            line=dict(dash="dot"),
        )
    )
    fig.update_layout(
        title="% Change (Normalized)", template="plotly_white", hovermode="x unified"
    )

    return fig


def plot_twinx(df: pl.DataFrame):
    ts = df["UTC-Datetime"].to_numpy()
    es_spread = df["Market Basis"].to_numpy()
    theo_spread = df["Theoretical Basis"].to_numpy()
    btic = df["BTIC"].to_numpy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=ts, y=es_spread, mode="lines", name="ES - Spot"), secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=ts, y=theo_spread, mode="lines", name="Theo - Spot"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=ts, y=btic, mode="lines", name="BTIC", line=dict(dash="dot")),
        secondary_y=True,
    )

    fig.update_layout(
        title="Spread Comparison: ES/Theo vs BTIC (Dual Axis)",
        xaxis_title="Timestamp",
        yaxis_title="Spread to Spot",
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Spread to Spot", secondary_y=False)
    fig.update_yaxes(title_text="BTIC", secondary_y=True)

    return fig


def plot_twinx_for_date(df, date_str):
    target_date = date.fromisoformat(date_str)
    subset = df.filter(pl.col("UTC-Datetime").dt.date() == target_date)
    if subset.is_empty():
        print(f"No data for {date_str}")
        return
    ts = subset["UTC-Datetime"].to_numpy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=ts, y=subset["Market Basis"].to_numpy(), name="ES - Spot"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=ts, y=subset["Theoretical Basis"].to_numpy(), name="Theo - Spot"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ts, y=subset["BTIC"].to_numpy(), name="BTIC", line=dict(dash="dot")
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Twinx Plot on {date_str}",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig
