import numpy as np
import pandas as pd


class BasisBacktester:
    def __init__(
        self,
        spot_raw_df,
        futures_raw_df,
        theoretical_model,
        price_col="Last",
        timestamp_col="Date-Time",
        dividend_col="Dividend",
        interval="1min",
    ):
        """
        spot_raw_df/futures_raw_df  : raw DataFrame from data source
        theoretical_model           : model used to produce theoretical basis
        price_col                   : price column
        timestamp_col               : timestamp column
        dividend_col                : dividend column (for spot)
        interval                    : resample interval, e.g. '1min', '15min'
        """
        self.spot_raw = spot_raw_df
        self.fut_raw = futures_raw_df
        self.theo_model = theoretical_model
        self.price_col = price_col
        self.timestamp_col = timestamp_col
        self.dividend_col = dividend_col
        self.interval = interval

        self.data = None
        self.trade_df = None
        self._preprocess()

    @staticmethod
    def clean_market_data(df, price_col="Last", timestamp_col="Date-Time"):
        """
        Clean raw market data
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        df["datetime"] = df[timestamp_col].dt.tz_convert("America/New_York")
        df = df[["datetime", price_col]].rename(columns={price_col: "price"})
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def resample_series(sr, interval, method="last"):
        if method == "ohlc":
            return sr.resample(interval).ohlc()
        else:
            return sr.resample(interval).agg(method)

    def _preprocess(self):
        spot = self.clean_market_data(self.spot_raw, self.price_col, self.timestamp_col)
        fut = self.clean_market_data(self.fut_raw, self.price_col, self.timestamp_col)

        spot = self.resample_series(spot["price"], self.interval, "last")
        fut = self.resample_series(fut["price"], self.interval, "last")

        spot_dvds = self.resample_series(
            spot[self.dividend_col], self.interval, "sum"
        ).fillna(0)
        spot_cum_dvds = spot_dvds.cumsum()  # Cumulative dividends

        df = pd.DataFrame(
            {"spot": spot, "futures": fut, "dividends": spot_cum_dvds}
        ).dropna()

        df["basis"] = df["futures"] - df["spot"]
        df["theoretical_basis"] = self.theo_model.get_theo_basis(spot)  # TODO
        self.data = df

    def run_backtest(
        self,
        long_threshold,
        short_threshold,
        commission_rate,
        market_impact,
        trade_size=1,
        position_limit=1,
    ):
        """
        long_threshold              : threshold to long basis (negative)
        short_threshold             : threshold to short basis (positive)
        commission_rate             : commission / amount traded
        market_impact               : impact on trade price
        trade_size                  : size per trade
        position_limit              : maximum position in either direction
        """
        trades = []
        df = self.data
        basis = df["basis"]
        upper_limit = df["theoretical_basis"] + short_threshold
        lower_limit = df["theoretical_basis"] + long_threshold

        # Generate signals
        signal = pd.Series(0, index=df.index)
        signal[basis > upper_limit] = -1  # signal to short basis
        signal[basis < lower_limit] = 1  # signal to long basis

        position_size = 0

        for t, sig in signal.items():
            if sig == 0:
                continue

            spot_price = df.loc[t, "spot"]
            fut_price = df.loc[t, "futures"]
            dividends = df.loc[t, "dividends"]
            spot_trade = {"timestamp": t, "instrument": "spot"}
            fut_trade = {"timestamp": t, "instrument": "futures"}

            if sig == -1 and position_size > -position_limit:  # buy spot, sell futures
                spot_trade["direction"] = "buy"
                fut_trade["direction"] = "sell"
                spot_trade["price"] = spot_price * (1 + market_impact)
                fut_trade["price"] = fut_price * (1 - market_impact)
                spot_trade["size"] = trade_size
                fut_trade["size"] = trade_size
                position_size -= trade_size
            elif sig == 1 and position_size < position_limit:  # sell spot, buy futures
                spot_trade["direction"] = "sell"
                fut_trade["direction"] = "buy"
                spot_trade["price"] = spot_price * (1 - market_impact)
                fut_trade["price"] = fut_price * (1 + market_impact)
                spot_trade["size"] = trade_size
                fut_trade["size"] = trade_size
                position_size += trade_size

            if spot_trade.get("size"):
                spot_trade["transaction_cost"] = (
                    spot_trade["price"] * spot_trade["size"] * commission_rate
                )
                fut_trade["transaction_cost"] = (
                    fut_trade["price"] * fut_trade["size"] * commission_rate
                )
                spot_trade["dividends"] = dividends
                trades.append(spot_trade)
                trades.append(fut_trade)

        if position_size != 0:  # Handle remaining positions
            t = df.index[-1]
            spot_price = df.loc[t, "spot"]
            fut_price = df.loc[t, "futures"]
            dividends = df.loc[t, "dividends"]
            spot_trade = {"timestamp": t, "instrument": "spot"}
            fut_trade = {"timestamp": t, "instrument": "futures"}
            if position_size > 0:
                spot_trade["direction"] = "buy"
                fut_trade["direction"] = "sell"
                spot_trade["price"] = spot_price * (1 + market_impact)
                fut_trade["price"] = fut_price * (1 - market_impact)
            else:
                spot_trade["direction"] = "sell"
                fut_trade["direction"] = "buy"
                spot_trade["price"] = spot_price * (1 - market_impact)
                fut_trade["price"] = fut_price * (1 + market_impact)
            spot_trade["size"] = abs(position_size)
            fut_trade["size"] = abs(position_size)
            spot_trade["transaction_cost"] = (
                spot_trade["price"] * spot_trade["size"] * commission_rate
            )
            fut_trade["transaction_cost"] = (
                fut_trade["price"] * fut_trade["size"] * commission_rate
            )
            spot_trade["dividends"] = dividends
            trades.append(spot_trade)
            trades.append(fut_trade)

        self.trades_df = pd.DataFrame(trades).set_index("timestamp").sort_index()

    def analyze_results(self):
        """Post trades analysis"""
        df = self.trades_df
        df["dividends"].fillna(0, inplace=True)
        sign = df["direction"].map({"buy": -1, "sell": 1})
        n_trades = int(len(df) / 2)
        total_pnl = (
            sign * (df["price"] + df["dividends"]) * df["size"] - df["transaction_cost"]
        ).cumsum()

        spot_trades = []
        holding_time = []
        trade_sizes = []
        for t, row in df[df["instrument"] == "spot"].iterrows():
            if not spot_trades or spot_trades[-1][1] == row["direction"]:
                spot_trades.append((t, row["direction"], row["size"]))
            else:
                last_trade = spot_trades.pop()
                last_trade_size = last_trade[2]
                holding_time.append((t - last_trade[0]).total_seconds() / 60)
                if last_trade_size < row["size"]:
                    trade_sizes.append(last_trade_size)
                    spot_trades.append(
                        (t, row["direction"], row["size"] - last_trade_size)
                    )
                elif last_trade_size > row["size"]:
                    trade_sizes.append(row["size"])
                    spot_trades.append(
                        (last_trade[0], last_trade[1], last_trade_size - row["size"])
                    )
                else:
                    trade_sizes.append(row["size"])

        avg_holding_time = np.average(
            holding_time, weights=trade_sizes
        )  # Size-weighted average holding time

        return {
            "n_trades": n_trades,
            "total_pnl": total_pnl,
            "avg_holding_time": avg_holding_time,
        }
