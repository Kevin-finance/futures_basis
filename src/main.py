# main.py
from datetime import datetime
import pytz
import polars as pl
import pandas as pd
from pathlib import Path

from settings import config
from expiration_calendar import ExpirationCalendar
from preprocessing import Preprocessor
import utils

def load_all_data():
    MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
    calendar = ExpirationCalendar(contract_type="es")

    # Assigning path to data
    sofr_path = Path(MANUAL_DATA_DIR / "SOFR_partial.csv")
    div_path = Path(MANUAL_DATA_DIR / "SPXDIV.csv")

    # divfut_trade_path = Path(
    #     MANUAL_DATA_DIR / "PJ_Lab_SPX_DIV_Trade_03_10_03_14_1min.csv.gz"
    # )
    divfut_quote_path = Path(
        MANUAL_DATA_DIR / "ES_DIV_1yr_Quote.csv.gz"
    )

    spx_path = Path(MANUAL_DATA_DIR / "SPX_1min.csv")

    es_trade_path = Path(MANUAL_DATA_DIR / "ES_1yr_Trade.csv.gz")

    btic_trade_path = Path(MANUAL_DATA_DIR / "BTIC_1yr_Trade.csv.gz")

    # Preprocessor instances for all data
    sofr = Preprocessor(sofr_path, calendar,
                        interpolation_method="pwc",
                        timezone="America/Chicago", batch_time= 5)
    div = Preprocessor(div_path, calendar,
                       timezone="America/New_York", batch_time= 5)
    # In reality we should set as 10
    # divfut_trade = Preprocessor(dir=divfut_trade_path, calendar=calendar)
    divfut_quote = Preprocessor(dir=divfut_quote_path, calendar=calendar)

    spx = Preprocessor(dir=spx_path, calendar=calendar)
    es_trade = Preprocessor(dir=es_trade_path, calendar=calendar)
    btic_trade = Preprocessor(dir=btic_trade_path, calendar=calendar)

    # Still sticking to lazyframe / df is lazyframes
    sofr_df = sofr.run_all()
    div_df = div.run_all2()
    div_fut_quote_df = divfut_quote.timezone_str_convert().get()

    spx_df = spx.timezone_str_convert2().get()

    es_trade_df = es_trade.timezone_str_convert().get()

    btic_trade_df = btic_trade.timezone_str_convert().get()

    return sofr_df, div_df, spx_df, es_trade_df, btic_trade_df , div_fut_quote_df

def prepare_merged(sofr_df, div_df, spx_df, es_df, btic_df, div_fut_quote_df):

    # Still sticking to lazyframe / df is lazyframes
    sofr_df = sofr_df.unique(subset=['UTC-Datetime']).sort("UTC-Datetime")

    # Calculating Mid-Price for dividend futures data
    
    div_fut_quote_df = div_fut_quote_df.with_columns(
        ((pl.col("Bid Price") + pl.col("Ask Price")) * 0.5).alias("Mid Price")
    )

    join_df = div_fut_quote_df.join_asof(
        div_df, left_on="UTC-Datetime", right_on="UTC-Datetime", strategy="backward"
    )

    join_df = join_df.with_columns(
        (pl.col("Mid Price") - pl.col("SPXDIV Index")).alias("Expected Points")
    )

    final_div = join_df.filter(pl.col("Expected Points").is_not_null())


    # Filters out whatever is out of start, end range
    start = pytz.UTC.localize(datetime(2025,3,10))
    end   = pytz.UTC.localize(datetime(2025,3,14))

    merged = utils.merge_and_filter(
        [spx_df.select(["UTC-Datetime","Close"]),
         sofr_df.select(["UTC-Datetime","Near_Rate","NearMonth_Years"]),
         div_df.select(["UTC-Datetime","SPXDIV Index"]),
         es_df.select(["UTC-Datetime","Last"]),
         btic_df.select(["UTC-Datetime","Last"]),
         final_div.select(["UTC-Datetime", "Expected Points"])],
        on="UTC-Datetime", strategy="backward", tolerance=None,
        start=start, end=end
    )

    # Compute Theoretical price and finalize columns to be displayed
    merged = utils.compute_adjustments(merged)
    merged = utils.compute_theoretical_prices(merged)
    merged = utils.finalize_columns(merged)

    return merged

def main():
    sofr_df, div_df, spx_df, es_df, btic_df, div_fut_quote_df  = load_all_data()
    
    merged = prepare_merged(sofr_df, div_df, spx_df, es_df, btic_df, div_fut_quote_df).collect()
    
    merged_pd = merged.to_pandas()
    merged_pd["UTC-Datetime"] = pd.to_datetime(merged_pd["UTC-Datetime"])
    utils.plot_basis(merged_pd).show()
    utils.plot_basis(merged_pd,in_bps=True).show()
    utils.plot_basis_dual(merged_pd).show()
    utils.plot_basis_dual(merged_pd,in_bps=True).show()

if __name__ == "__main__":
    main()
    