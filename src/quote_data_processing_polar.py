import polars as pl
import numpy as np
import plotly.graph_objects as go


class TopOfBookTheoWeighted:
    """
    Compute theoretical price at top of book weighted by bid and ask sizes.
    """
    def __init__(self, bid_col: str = 'Bid Price',
                 ask_col: str = 'Ask Price',
                 bid_size_col: str = 'Bid Size',
                 ask_size_col: str = 'Ask Size'):
        self.bid_col = bid_col
        self.ask_col = ask_col
        self.bid_size_col = bid_size_col
        self.ask_size_col = ask_size_col

    def compute(self, df: pl.DataFrame) -> pl.Series:
        bid = df[self.bid_col]
        ask = df[self.ask_col]
        bid_size = df[self.bid_size_col]
        ask_size = df[self.ask_size_col]
        total_size = bid_size + ask_size
        theo = (bid * bid_size + ask * ask_size) / total_size
        return theo


class TopOfBookTheoMid:
    """
    Compute theoretical price at top of book as mid-price.
    """
    def __init__(self, bid_col: str = 'Bid Price', ask_col: str = 'Ask Price'):
        self.bid_col = bid_col
        self.ask_col = ask_col

    def compute(self, df: pl.DataFrame) -> pl.Series:
        return (df[self.bid_col] + df[self.ask_col]) / 2


def hampel_filter(df: pl.DataFrame,
                  column: str,
                  window: int = 10,
                  threshold: float = 3.0) -> pl.DataFrame:
    """
    Apply Hampel filter to a numeric column in a Polars DataFrame.
    Adds a new column `<column>_filtered` with outliers replaced by median.
    """
    values = df[column].to_numpy()
    filtered = values.copy()
    half = window // 2

    for i in range(half, len(values) - half):
        window_slice = values[i - half:i + half + 1]
        median = np.median(window_slice)
        mad = np.median(np.abs(window_slice - median))
        mad = mad if mad != 0 else 1e-6
        if abs(values[i] - median) > threshold * mad:
            filtered[i] = median

    return df.with_columns(
        pl.Series(name=f"{column}_filtered", values=filtered)
    )


def analyze_and_plot(df: pl.DataFrame,
                     model,
                     timestamp_col: str = 'Date-Time',
                     window: int = 10,
                     threshold: float = 3.0) -> None:
    """
    Compute theoretical values using the given model, apply Hampel filter,
    print error metrics and plot original vs filtered theoretical price.
    """
    # Ensure datetime sorting
    df = df.with_columns([
        pl.col(timestamp_col).str.to_datetime(strict=False)
    ]).sort(timestamp_col)

    # Compute theoretical price
    theo = model.compute(df)
    df = df.with_columns([
        theo.alias("model_value"),
        theo.alias("model_value_original")
    ])

    # Apply Hampel filter
    df = hampel_filter(df, "model_value", window=window, threshold=threshold)

    # Error metrics
    original = df["model_value_original"].to_numpy()
    filtered = df["model_value_filtered"].to_numpy()
    valid_idx = (
        np.isfinite(original[1:]) &
        np.isfinite(original[:-1]) &
        (original[:-1] != 0)
    )
    error_term_original = np.log(original[1:][valid_idx] / original[:-1][valid_idx])
    error_term_filtered = np.log(filtered[1:][valid_idx] / filtered[:-1][valid_idx])

    original_error_mean = np.mean(error_term_original)
    filtered_error_mean = np.mean(error_term_filtered)
    original_mae = np.mean(np.abs(error_term_original))
    filtered_mae = np.mean(np.abs(error_term_filtered))

    print(f"Original Error Mean: {original_error_mean:.10f}")
    print(f"Filtered Error Mean: {filtered_error_mean:.10f}")
    print(f"Original MAE: {original_mae:.8f}")
    print(f"Filtered MAE: {filtered_mae:.8f}")

    # Plotting
    timestamps = df[timestamp_col].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=original,
        mode='lines',
        name='Original Theo Price'
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=filtered,
        mode='lines',
        name='Filtered Theo Price'
    ))
    fig.update_layout(
        title=f"Theoretical Price (Model: {model.__class__.__name__})",
        xaxis_title="Timestamp",
        yaxis_title="Theo Price",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified"
    )
    fig.show()
