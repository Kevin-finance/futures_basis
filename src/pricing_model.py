from abc import ABC
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

from expiration_calendar import ExpirationCalendar
from preprocessing import Preprocessor


class PricingModel(ABC):
    """Base class for pricing models."""
    pass


class CostOfCarryModel(PricingModel):
    def __init__(
        self,
        sofr_path: Path,
        div_index_path: Path,
        div_futures_path: Path,
        spot_path: Path,
        futures_path: Path,
        btic_path: Path,
        calendar: ExpirationCalendar,
        **kwargs
    ):
        self.sofr_path = sofr_path
        self.div_index_path = div_index_path
        self.div_futures_path = div_futures_path
        self.spot_path = spot_path
        self.futures_path = futures_path
        self.btic_path = btic_path
        self.calendar = calendar
        self.config = kwargs

        self.data = {}
        self.output = None

    def _preprocess(self):
        calendar = self.calendar
        config = self.config

        sofr_pp = Preprocessor(
            self.sofr_path,
            calendar,
            interpolation_method=config.get("sofr_interpolation_method", "pwc"),
            timezone=config.get("sofr_timezone", "America/Chicago"),
            batch_time=config.get("sofr_batch_time", 5)
        )
        div_index_pp = Preprocessor(
            self.div_index_path, 
            calendar, 
            timezone=config.get("div_index_timezone", "America/New_York"), 
            batch_time=config.get("div_index_batch_time", 19)
        )
        div_futures_pp = Preprocessor(self.div_futures_path, calendar)
        spot_pp = Preprocessor(self.spot_path, calendar)
        futures_pp = Preprocessor(self.futures_path, calendar)
        btic_pp = Preprocessor(self.btic_path, calendar)

        self.data['sofr'] = sofr_pp.run_all().collect().unique(subset=["Date"], keep="first").sort("Date")
        self.data['div_index'] = div_index_pp.run_all2().collect()
        self.data['div_futures'] = div_futures_pp.timezone_str_convert().get().collect()
        self.data['spot'] = spot_pp.timezone_str_convert2().get().collect()
        self.data['futures'] = futures_pp.timezone_str_convert().get().collect()
        self.data['btic'] = btic_pp.timezone_str_convert().get().collect()

        self.data['div_futures'] = self.data['div_futures'].with_columns(
            ((pl.col("Bid Price") + pl.col("Ask Price")) * 0.5).alias("Mid Price")
        )
    
    @staticmethod
    def merge_asof_tables(
        *dfs, on: str, strategy: str = "backward", tolerance: str = None
    ):
        def join_func(left, right):
            if tolerance:
                return left.join_asof(
                    right, on=on, strategy=strategy, tolerance=tolerance
                )
            else:
                return left.join_asof(right, on=on, strategy=strategy)

        return reduce(join_func, dfs)
    
    def price_basis(self, start: datetime, end: datetime):
        self._preprocess()
        data = self.data
        div_join_df = data['div_futures'].join_asof(
            data['div_index'], left_on="UTC-Datetime", right_on="UTC-Datetime", strategy="backward"
        )
        div_join_df = div_join_df.with_columns(
            (pl.col("Mid Price") - pl.col("SPXDIV Index")).alias("Expected Points")
        )
        div_final_df = div_join_df.filter(pl.col("Expected Points").is_not_null()).select(
            ["UTC-Datetime", "Expected Points"]
        )

        merged = self.merge_asof_tables(
            data['spot'].select(["UTC-Datetime", "Close"]),
            data['sofr'].select(["UTC-Datetime", "Near_Rate", "NearMonth_Years"]),
            div_final_df,
            data['futures'].select(["UTC-Datetime", "Last"]),
            data['btic'].select(["UTC-Datetime", "Last"]),
            on="UTC-Datetime",
            strategy="backward"
        )

        merged_filtered = merged.filter(
            (pl.col("UTC-Datetime") >= start) & (pl.col("UTC-Datetime") <= end)
        )
        merged_filtered = merged_filtered.with_columns(
            (
                (pl.col("Close") - pl.col("Expected Points"))
                * np.exp((pl.col("Near_Rate") / 100) * pl.col("NearMonth_Years"))
            ).alias("Theoretical Futures Price")
        )
        merged_filtered = merged_filtered.rename(
            {
                "Close": "Spot",
                "Expected Points": "Expected Dividend Points",
                "Last": "ES Trade",
                "Last_right": "BTIC",
            }
        )
        merged_filtered = merged_filtered.with_columns(
            (pl.col("Theoretical Futures Price") - pl.col("Spot")).alias(
                "Theoretical Basis"
            )
        )
        merged_filtered = merged_filtered.with_columns(
            (pl.col("ES Trade") - pl.col("Spot")).alias("Market Basis")
        )
        merged_filtered = merged_filtered.with_columns(
            ((pl.col("Theoretical Basis")*10000)/pl.col("Spot")).alias("Theoretical Basis Rate")
        )
        merged_filtered = merged_filtered.with_columns(
            ((pl.col("Market Basis")*10000)/pl.col("Spot")).alias("Market Basis Rate")
        )
        merged_filtered = merged_filtered.with_columns(
            ((pl.col("BTIC")*10000)/pl.col("Spot")).alias("BTIC Rate")
        )
        self.output = merged_filtered

        return merged_filtered.to_pandas()

    def generate_output_plot(self, in_bps: bool = False):
        if self.output is None:
            return
        output_pd = self.output.to_pandas()
        output_pd["UTC-Datetime"] = pd.to_datetime(
            output_pd["UTC-Datetime"]
        )
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
 
        