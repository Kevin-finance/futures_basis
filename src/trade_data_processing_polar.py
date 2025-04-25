import polars as pl
import numpy as np

class RollingVWAPModel:
    """
    Compute rolling VWAP (Volume-Weighted Average Price) over a fixed window.
    """
    def __init__(self, price_col: str = 'Price', volume_col: str = 'Volume', window: int = 5):
        self.price_col = price_col
        self.volume_col = volume_col
        self.window = window

    def compute(self, df: pl.DataFrame) -> pl.Series:
        # Compute dollar volume and rolling sums
        dollar_volume = df[self.price_col] * df[self.volume_col]
        num = dollar_volume.rolling_sum(window_size=self.window, min_periods=1)
        den = df[self.volume_col].rolling_sum(window_size=self.window, min_periods=1)
        return (num / den).alias(f"vwap_{self.window}")


def hampel_filter(
    df: pl.DataFrame,
    column: str,
    window: int = 10,
    threshold: float = 3.0
) -> pl.DataFrame:
    """
    Applies a Hampel filter to smooth a time series by replacing outliers
    with the local median. Returns the DataFrame with a new column:
      <column>_filtered
    """
    half = window // 2
    vals = df[column].to_numpy()
    filtered = vals.copy()
    outliers = 0

    # Slide over the series and replace outliers
    for i in range(half, len(vals) - half):
        window_vals = vals[i - half: i + half + 1]
        med = np.median(window_vals)
        mad = np.median(np.abs(window_vals - med))
        # Avoid zero MAD
        mad = mad if mad != 0 else 1e-9
        if abs(vals[i] - med) > threshold * mad:
            filtered[i] = med
            outliers += 1

    print(f"Hampel filter replaced {outliers} outliers in '{column}'.")
    return df.with_columns(
        pl.Series(name=f"{column}_filtered", values=filtered)
    )


def process_trade_data(
    df: pl.DataFrame,
    price_col: str = 'Last',
    volume_col: str = 'Volume',
    timestamp_col: str = 'Date-Time',
    vwap_window: int = 5,
    filter_window: int = 10,
    filter_threshold: float = 3.0
) -> pl.DataFrame:
    """
    Full pipeline to process trade data:
      1. Sort by timestamp_col
      2. Compute rolling VWAP
      3. Apply Hampel filter to the VWAP series
    Returns the augmented DataFrame.
    """
    # Ensure correct types and ordering
    df = (
        df.with_columns(
            pl.col(timestamp_col).str.to_datetime(strict=False)
        )
        .sort(timestamp_col)
    )

    # Compute VWAP
    vwap_model = RollingVWAPModel(
        price_col=price_col,
        volume_col=volume_col,
        window=vwap_window
    )
    df = df.with_columns(vwap_model.compute(df))

    # Smooth VWAP with Hampel filter
    df = hampel_filter(
        df,
        column=f"vwap_{vwap_window}",
        window=filter_window,
        threshold=filter_threshold
    )

    return df


# data reading is handled outside

