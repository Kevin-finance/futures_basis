from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytz

import interest_rate
from expiration_calendar import ExpirationCalendar
from settings import config



class Preprocessor:
    def __init__(
        self,
        dir,
        calendar: ExpirationCalendar,
        model_type: str = "term_structure",
        interpolation_method: str = None,
        timezone: str = None,
        batch_time: int = None,
    ):
        # Polars new update - memory management, 
        # this guarantees the query does not loads entire data on memory rather loads chunks so parallelization is achieved
        pl.Config.set_engine_affinity(engine='streaming')

        # timezone parameter is for timezone you want to declare your daily data as.
        # America/Chicago for CT , America/New_York for ET
        # batch_time is the time when the data is published. It is used to add that hour data to daily data so that we are not looking forward
        """
        Expects dataframe as below and saved in self.df

        - Date: datetime[ns]
        - TSFR1M Index: float64
        - TSFR3M Index: float64
        - TSFR6M Index: float64
        - TSFR12M Index: float64

        These tenor corresponds to Term SOFR calculated by CME
        This dataframe is in wide format

        Calendar instance is needed to calculate remaining maturity until the near, far month futures expiry as of the date

        We can also set interpolation method to interest rate, passed as an instance

        """

        self.df = self._load_data(Path(dir))
        self.calendar = calendar  # e.g., ExpirationCalendar('es')

        self.model_type = model_type
        self.interpolation_method = interpolation_method

        self.interest_rate_model = None

        self.timezone = timezone
        self.batch_time = batch_time

    def build_interest_rate_model(self, params: dict):
        """
        params:
        - if flat: {"rate": float}
        - if term_structure: {"times": array, "values": array}
        """

        if self.model_type == "flat":
            self.interest_rate_model = interest_rate.get_interest_rate_model(
                model_type="flat", params=params
            )
        elif self.model_type == "term_structure":
            if self.interpolation_method is None:
                raise ValueError(
                    "Interpolation method must be provided for term structure model."
                )
            self.interest_rate_model = interest_rate.get_interest_rate_model(
                model_type="term_structure",
                params={
                    "times": params["times"],
                    "values": params["values"],
                    "interpolation_method": self.interpolation_method,
                },
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return self

    def _load_data(self, path: Path):
        # Instead of reading it right away on memory(eager execution), waits until query (lazyframe)
        suffixes = [s.lower() for s in path.suffixes]
        if ".csv" in suffixes:
            self.df = pl.scan_csv(path, try_parse_dates=False)

        elif ".parquet" in suffixes:
            self.df = pl.scan_parquet(path)

        else:
            raise ValueError("File format not recognized")
        return self.df

    def check_data_type(self):
        """
        This converts Date object to datetime and converts #NA to None which is a typical null in bbg data
        """
        self.df = self.df.with_columns(
            pl.col("Date")
            .str.strptime(pl.Datetime, format="%m/%d/%Y")
            .dt.cast_time_unit("ns")
        )

        self.df = self.df.with_columns(
            [
                pl.when(pl.col(col) != "#N/A")
                .then(pl.col(col))
                .otherwise(None)
                .cast(pl.Float64)
                .alias(col)
                for col, dtype in self.df.collect_schema().items()
                if col != "Date" and dtype == pl.Utf8
            ]
        )

        return self

    def compute_remain_ttm(self):
        """
        This method calculates remaining maturity until the near month and far month in days and years
        FOR FUTURES
        """
        # Checks minimum date and maximum date and save it as a variable(datetime ns)
        start_date = (
            self.df.select(pl.col("Date").min()).collect().item().strftime("%Y-%m-%d")
        )
        end_date = (
            self.df.select(pl.col("Date").max()).collect().item().strftime("%Y-%m-%d")
        )

        # Utilize the expiration_calendar.py
        expirations = self.calendar.get_expiration(
            start_date, end_date
        )  # returns DateTimeIndex
        expirations = pd.to_datetime(expirations)

        dates = self.df.select("Date").collect().to_pandas()["Date"]

        # For each dates you run a loop, filters expiration that is after than current date and find first two
        # Subtract current date from the corresponding maturity
        if self.calendar.contract_forward == 2:
            near_days = []
            far_days = []
            for val_date in dates:
                expiries_after = expirations[expirations > val_date]
                first = expiries_after[0]
                second = expiries_after[1]
                near_days.append((first - val_date).days)
                far_days.append((second - val_date).days)

            # Putting together in polars again
            self.df = self.df.with_columns(
                [
                    pl.Series(name="NearMonth_Days", values=near_days),
                    pl.Series(name="FarMonth_Days", values=far_days),
                    (pl.Series(name="NearMonth_Days", values=near_days) / 365).alias(
                        "NearMonth_Years"
                    ),
                    (pl.Series(name="FarMonth_Days", values=far_days) / 365).alias(
                        "FarMonth_Years"
                    ),
                ]
            )
        else:
            raise ValueError("Currently contract forward over 2 not supported")

        return self

    def compute_sofr_years(self):
        """
        Convert Term sofr to years e.g 1D-> 1/365 years
        """
        # This should be changed to 1M , 3M , 6M , 12M , once we have the data
        sofr_columns = ["TSFR1M Index", "TSFR3M Index", "TSFR6M Index", "TSFR12M Index"]

        index_columns = [
            "Date",
            "NearMonth_Days",
            "FarMonth_Days",
            "NearMonth_Years",
            "FarMonth_Years",
        ]
        self.df = (
            self.df.unpivot(index=index_columns, on=sofr_columns)
            .rename({"variable": "Tenor", "value": "Rate"})
            .sort(["Date"])
        )
        # ensure polars casts Rate as float
        self.df = self.df.with_columns(pl.col("Rate").cast(pl.Float64))
        self.df = self.df.with_columns(
            [
                pl.col("Tenor")
                .map_elements(lambda t: self.parse_dates(t), return_dtype=pl.Int64)
                .alias("Reference_Date")
            ]
        )
        self.df = self.df.with_columns(
            [(pl.col("Reference_Date") / 365).alias("Reference_Year")]
        )

        convert_rate = 365/360
        self.df = self.df.with_columns(
            [
                pl.when(pl.col("Rate").is_not_null())
                .then(pl.col("Rate") * convert_rate)
                .otherwise(None)
            ]
        )
        return self

    def parse_dates(self, tenor: str) -> int:
        # Converts Term Sofr literal into days
        # e.g. 1M - >30 days
        tenor_clean = tenor.replace("TSFR", "").replace("Index", "").strip()

        num, dt = "", ""
        for char in tenor_clean:
            if char.isdigit():
                num += char
            elif char.isalpha():
                dt += char

        num = int(num)

        if dt == "D":
            num *= 1
        elif dt == "W":
            num *= 7
        elif dt == "M":
            num *= 30
        elif dt == "Y":
            num *= 365
        else:
            raise ValueError(f"Unsupported date type: {dt}")

        return num

    def align_sofr_curve(self):
        """align rates and take care of null values
        remove rows with Rates having null values
        We need this process so that we are not putting null values in the interpolator"""

        self.df = self.df.filter(pl.col("Rate").is_not_null())
        self.df = self.df.sort(by=["Date", "Reference_Date"], descending=[False, False])

        return self

    def build_term_structure(self):
        """
        This produces near month and far month interest rate corresponding to each tenor
        by running interpolation we set.


        """
        df = (
            self.df.collect().to_pandas()
        )  # LazyFrame → eager DataFrame → pandas conversion

        results = []
        for date, group in df.groupby("Date"):
            x = group["Reference_Year"].to_numpy()
            y = group["Rate"].to_numpy()

            near = group["NearMonth_Years"].to_numpy()[0]
            far = group["FarMonth_Years"].to_numpy()[0]

            self.build_interest_rate_model(params={"times": x, "values": y})
            near_rate = self.interest_rate_model.get_rate(near)
            far_rate = self.interest_rate_model.get_rate(far)

            results.append({"Date": date, "Near_Rate": near_rate, "Far_Rate": far_rate})

        result_df = pl.DataFrame(results).with_columns(
            [
                pl.col("Near_Rate").cast(pl.Float64),
                pl.col("Far_Rate").cast(pl.Float64),
            ]
        )
        result_df = result_df.with_columns(
            pl.col("Date").cast(pl.Datetime("ns"))
        ).lazy()

        self.df = self.df.join(result_df, on="Date", how="left")

        return self
    
    ###NEW####
    # def convert_to_continuous_compounding_rates(self):
    #     """
    #     After interpolated Near_Rate and Far_Rate are created,
    #     convert them from simple interest (ACT/365) to continuous compounding.
    #     Uses: ln(1 + r * T)
    #     """

    #     self.df = self.df.with_columns([
    #         (pl.col("Near_Rate") / 100).alias("Near_Rate_Decimal"),
    #         (pl.col("Far_Rate") / 100).alias("Far_Rate_Decimal"),
    #     ])

    #     self.df = self.df.with_columns([
    #         (1 + pl.col("Near_Rate_Decimal") * pl.col("NearMonth_Years"))
    #         .log()
    #         .alias("Near_Rate_CC"),

    #         (1 + pl.col("Far_Rate_Decimal") * pl.col("FarMonth_Years"))
    #         .log()
    #         .alias("Far_Rate_CC"),
    #     ])

    #     return self
    
    def convert_to_utc(self):
        """
        For Bloomberg and data and daily data in general, they are not timezone aware
        Thus we need to use the timezone information to convert it into UTC for better comparison
        """
        # First convert into Datetime from date
        # self.df = self.df.with_columns(pl.col("Date").cast(pl.Datetime(time_unit='ns')))

        # Assigns timezone that was put in as input, and then converts it into UTC
        self.df = self.df.with_columns(
            pl.col("Date")
            .dt.replace_time_zone(self.timezone)
            .dt.convert_time_zone("UTC")
            .cast(pl.Datetime("ns", time_zone="UTC"))
            .alias("UTC-Datetime")
        )
        return self

    def add_time_info(self):
        """ "
        This methods adds time information for daily data so that we are not looking forward
        e.g. on 4/11 data we are adding 05:00AM so that when we perform join_asof we are using present data
        
        Need to account for publish / distributed time
        Dividend index e.g 1/17 data is distributed a day before
        """

        # For dividend index data
        if not isinstance(self.df.collect_schema()["Date"], pl.Datetime):
            self.df = self.df.with_columns(
                pl.col("Date").str.strptime(pl.Datetime, format="%m/%d/%Y")
            )

        self.df = self.df.with_columns(
            (
                pl.datetime(
                    year=pl.col("Date").dt.year(),
                    month=pl.col("Date").dt.month(),
                    day=pl.col("Date").dt.day(),
                    hour=self.batch_time,  # ex: 5
                )
                .dt.replace_time_zone(self.timezone)
                .alias("Date")
            )
        )

        return self


    def timezone_str_convert(self):
        # This is to parse Prime Trading provided data date to timezone aware datetime

        self.df = self.df.with_columns(
            (
                pl.col("Date-Time").str.strptime(
                    pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%9fZ", strict=True
                )
            )
            .dt.replace_time_zone("UTC")
            .alias("Date-Time")
        )
        self.df = self.df.rename({"Date-Time": "UTC-Datetime"})
        self.df = self.df.with_columns(
            (pl.col("UTC-Datetime") + pl.duration(hours=pl.col("GMT Offset")))
            .dt.replace_time_zone("America/Chicago")
            .alias("Date")
        )

        return self

    def timezone_str_convert2(self):
        """
        For SPX realtime data from barchart.com where input times are in America/New_York (naive)
        """
        # First converts timezone naive to ET
        self.df = self.df.with_columns(
            pl.col("Date")
            .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M", strict=True)
            .dt.replace_time_zone("America/New_York").alias("Date")
        )
        # Convert that ET to UTC time
        self.df = self.df.with_columns(pl.col("Date").dt.convert_time_zone("UTC")
                                      .cast(pl.Datetime("ns", time_zone="UTC"))
                                    .alias("UTC-Datetime")
        )
            
        return self

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


    def run_all(self):
        # pipeline for sofr
        return (
            self.check_data_type()
            .compute_remain_ttm()
            .compute_sofr_years()
            .align_sofr_curve()
            .build_term_structure() #.convert_to_continuous_compounding_rates()
            .add_time_info()
            .convert_to_utc()
            .get()
        )

    def run_all2(self):
        # for dividend index
        return self.add_time_info().convert_to_utc().get()

    def get(self) -> pl.DataFrame:
        return self.df


if __name__ == "__main__":
    # Memo
    # Check futures pricing formula
    # Check batch time
    # Div quote data very small
    # 1M - > 30day fixed
    # Continous compounding conversion
    # BTIC forward fill? 
    # maturity - not the whole day (until expiry)
    # Dividend index does not account for special dividend
    # interest rate should be calculated only until its expiry not at EOD
    print()
    
