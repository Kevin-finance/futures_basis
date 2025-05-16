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
from dateutil.relativedelta import relativedelta



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
        # This guarantees the query does not loads entire data on memory rather loads chunks so parallelization is achieved
        pl.Config.set_engine_affinity(engine='streaming')

        # Timezone parameter is for timezone you want to declare your daily data as.
        # America/Chicago for CT , America/New_York for ET
        # Batch_time is the time when the data is published. It is used to add that hour data to daily data so that we are not looking forward
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
    
    def remove_null(self):
        """
        This methods removes row that contains null values for every column
        """
        # pl.all().is_null() produces a series and masking only all True values
        self.df = self.df.filter(~pl.all_horizontal(pl.all().is_null()))
        return self

    def check_data_type(self):
        """
        This converts Date object to datetime and converts #NA to None 
        which is a typical null in bbg data
        """
        #Step1. Parses Date column in string to datetime object 
        self.df = self.df.with_columns(
            pl.col("Date")
            .str.strptime(pl.Datetime, format="%m/%d/%Y")
            .dt.cast_time_unit("ns")
        )

        #Step2. Ensuring each column is float type not a string due to #N/A values
        # If this process is that processed , then the column's dtype is set to string
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

    def parse_front_month(self):
        """
        This method parses only the front month contract for a given date
        """
        # A helper to “materialize” and print
        self.df = self.df.with_columns([
        pl.col("#RIC").str.strip_chars().alias("#RIC")
    ])

        # 1) Parse month_code & contract_year → contract_month & contract_expiry
        self.df = (
            self.df
            .with_columns([
                pl.when(pl.col("#RIC").str.len_chars() == 5)
                .then(pl.col("#RIC").str.slice(2, 1))
                .when(pl.col("#RIC").str.len_chars() == 6)
                .then(pl.col("#RIC").str.slice(3, 1))
                .otherwise(None)
                .alias("month_code"),
                (pl.col("#RIC")
                .str.extract(r"(\d{2})$", 1)
                .cast(pl.Int32)
                + 2000
                ).alias("contract_year")
            ])
            .with_columns([
                pl.when(pl.col("month_code")=="F").then(1)
                .when(pl.col("month_code")=="G").then(2)
                .when(pl.col("month_code")=="H").then(3)
                .when(pl.col("month_code")=="J").then(4)
                .when(pl.col("month_code")=="K").then(5)
                .when(pl.col("month_code")=="M").then(6)
                .when(pl.col("month_code")=="N").then(7)
                .when(pl.col("month_code")=="Q").then(8)
                .when(pl.col("month_code")=="U").then(9)
                .when(pl.col("month_code")=="V").then(10)
                .when(pl.col("month_code")=="X").then(11)
                .when(pl.col("month_code")=="Z").then(12)
                .otherwise(None)
                .alias("contract_month")
            ])
            .filter(pl.col("contract_month").is_not_null())
            .with_columns([
                pl.concat_str([
                    pl.col("contract_year").cast(pl.Utf8),
                    pl.lit("-"),
                    pl.col("contract_month").cast(pl.Utf8).str.zfill(2),
                    pl.lit("-01"),
                ])
                .str.strptime(pl.Date, "%Y-%m-%d")
                .cast(pl.Datetime("ns"))
                .dt.replace_time_zone("UTC")
                .alias("contract_expiry")
            ])
        )

        # 2) Build the calendar of real expiries
        start = self.df.select(pl.col("UTC-Datetime").min()).collect().item().strftime("%Y-%m-%d")
        end   = self.df.select(pl.col("UTC-Datetime").max()).collect().item().strftime("%Y-%m-%d")
        expiration = (
            pl.DataFrame({"expiry": self.calendar.get_expiration(start, end)})
            .sort("expiry")
            .lazy()
        )

        # 3) Forward-as-of join ON THE TIMESTAMP to tag each row with the next real expiry
        self.df = (
            self.df
            .sort("UTC-Datetime")
            .join_asof(
                expiration,
                left_on="UTC-Datetime",
                right_on="expiry",
                strategy="forward"
            )
            .filter(pl.col("expiry").is_not_null())
        )

        # 4) Keep only those rows whose parsed year/month equals the tagged expiry’s year/month
        self.df = self.df.filter(
            (pl.col("contract_expiry").dt.year()  == pl.col("expiry").dt.year()) &
            (pl.col("contract_expiry").dt.month() == pl.col("expiry").dt.month())
        )

        # 5) Deduplicate so one quote per timestamp
        self.df = (
            self.df
            .sort(["UTC-Datetime", "contract_expiry"])
            .unique(subset=["UTC-Datetime"], keep="first")
        )

        # 6) Drop helper columns
        self.df = self.df.drop([
            "month_code","contract_year","contract_month",
            "contract_expiry","expiry"
        ])

        return self
    # Now `filled` has one row per original timestamp, with the most
    # recent front‐month quote 
    #     self.df = self.df.with_columns([
    #     # 1) month_code
    #     pl.when(pl.col("#RIC").str.len_chars() == 5)
    #     .then(pl.col("#RIC").str.slice(2, 1))   # ESH25 → "H"
    #     .when(pl.col("#RIC").str.len_chars() == 6)
    #     .then(pl.col("#RIC").str.slice(3, 1))   # SDIZ24 → "Z"
    #     .otherwise(None)
    #     .alias("month_code"),

    #     # 2) contract_year
    #     (pl.col("#RIC")
    #     .str.extract(r"(\d{2})$", 1)
    #     .cast(pl.Int32)
    #     + 2000
    #     ).alias("contract_year")
    # ])

    #         # Month code mapping
    #         # New column contract_month mapping e.g H->3 and then filters out null rows
    #     self.df = self.df.with_columns([
    #                 pl.when(pl.col("month_code") == "F").then(1).when(pl.col("month_code") == "G").then(2)
    #                 .when(pl.col("month_code") == "H").then(3).when(pl.col("month_code") == "J").then(4)
    #                 .when(pl.col("month_code") == "K").then(5).when(pl.col("month_code") == "M").then(6)
    #                 .when(pl.col("month_code") == "N").then(7).when(pl.col("month_code") == "Q").then(8)
    #                 .when(pl.col("month_code") == "U").then(9).when(pl.col("month_code") == "V").then(10)
    #                 .when(pl.col("month_code") == "X").then(11).when(pl.col("month_code") == "Z").then(12)
    #                 .otherwise(None).alias("contract_month")
    #             ]).filter(pl.col("contract_month").is_not_null())

    #     self.df = self.df.with_columns([
    #             pl.concat_str([
    #                 pl.col("contract_year").cast(pl.Utf8),
    #                 pl.lit("-"),
    #                 pl.col("contract_month").cast(pl.Utf8).str.zfill(2),
    #                 pl.lit("-01")
    #             ]).str.strptime(pl.Date, "%Y-%m-%d")
    #             .cast(pl.Datetime("ns"))
    #             .dt.replace_time_zone("UTC")
    #             .alias("contract_expiry")
    #         ]).sort(by='contract_expiry')
        
    #     start = (
    #     self.df.select(pl.col("UTC-Datetime").min())
    #            .collect()
    #            .item()
    #            .strftime("%Y-%m-%d")
    # )
    #     end = (
    #         self.df.select(pl.col("UTC-Datetime").max())
    #             .collect()
    #             .item()
    #             .strftime("%Y-%m-%d")
    #     )
    #     expiration = (
    #         pl.DataFrame({"expiry": self.calendar.get_expiration(start, end)})
    #         .sort("expiry")
    #         .lazy()
    #     )

    #     # 5) forward asof join → 각 row에 “다음 만기(expiry)” 태그, null(=다음 만기 없으면) 드롭
    #     self.df = (
    #         self.df.sort("contract_expiry")
    #             .join_asof(
    #                     expiration,
    #                     left_on="contract_expiry",
    #                     right_on="expiry",
    #                     strategy="forward"
    #             )
    #             .filter(pl.col("expiry").is_not_null())
    #     )

    #     # 6) 같은 UTC-Datetime 내에서 가장 가까운 contract_expiry(=front month)만 남기기
    #     self.df = (
    #         self.df
    #         .sort(["UTC-Datetime", "contract_expiry"])
    #         .unique(subset=["UTC-Datetime"], keep="first")
    #     )

    #     # 7) helper 컬럼 제거
    #     self.df = self.df.drop([
    #         "month_code", "contract_year", "contract_month",
    #         "contract_expiry", "expiry"
    #     ])


    #     return self
        # div_fut parsed wrongly bc there exists timeframe that has no front month quote data 





        # ES Code successful
#         self.df = self.df.with_columns([
#     # month_code
#     pl.when(pl.col("#RIC").str.len_chars() == 5)
#       .then(pl.col("#RIC").str.slice(2, 1))  # ESH25
#       .when(pl.col("#RIC").str.len_chars() == 6)
#       .then(pl.col("#RIC").str.slice(3, 1))  # SDIH25
#       .otherwise(None)
#       .alias("month_code"),

#     # contract_year 
#     pl.when(pl.col("#RIC").str.len_chars() == 5)
#       .then(2000 + pl.col("#RIC").str.extract(r"(\d{2})$", 1).cast(pl.Int32))
#       .when(pl.col("#RIC").str.len_chars() == 6)
#       .then(2000 + pl.col("#RIC").str.extract(r"(\d{2})$", 1).cast(pl.Int32))
#       .otherwise(None)
#       .alias("contract_year")
# ])

#         # Month code mapping
#         # New column contract_month mapping e.g H->3 and then filters out null rows
#         self.df = self.df.with_columns([
#                 pl.when(pl.col("month_code") == "F").then(1).when(pl.col("month_code") == "G").then(2)
#                 .when(pl.col("month_code") == "H").then(3).when(pl.col("month_code") == "J").then(4)
#                 .when(pl.col("month_code") == "K").then(5).when(pl.col("month_code") == "M").then(6)
#                 .when(pl.col("month_code") == "N").then(7).when(pl.col("month_code") == "Q").then(8)
#                 .when(pl.col("month_code") == "U").then(9).when(pl.col("month_code") == "V").then(10)
#                 .when(pl.col("month_code") == "X").then(11).when(pl.col("month_code") == "Z").then(12)
#                 .otherwise(None).alias("contract_month")
#             ]).filter(pl.col("contract_month").is_not_null())

#         self.df = self.df.with_columns([
#                 pl.concat_str([
#                     pl.col("contract_year").cast(pl.Utf8),
#                     pl.lit("-"),
#                     pl.col("contract_month").cast(pl.Utf8).str.zfill(2),
#                     pl.lit("-01")
#                 ]).str.strptime(pl.Date, "%Y-%m-%d")
#                 .cast(pl.Datetime("ns"))
#                 .dt.replace_time_zone("UTC")
#                 .alias("contract_expiry")
#             ]).sort(by='contract_expiry')
        
#         start = self.df.select(pl.col("UTC-Datetime").min()).collect().item().strftime("%Y-%m-%d")
#         end = self.df.select(pl.col("UTC-Datetime").max()).collect().item().strftime("%Y-%m-%d")
#         expiration = pl.DataFrame({"expiry":self.calendar.get_expiration(start,end)}).sort(by='expiry').lazy()
        
#         self.df = self.df.join_asof(expiration,left_on='contract_expiry',right_on='expiry',strategy='forward')
#         closest_expiry = self.df.group_by("UTC-Datetime").agg(pl.col("contract_expiry").min().alias("min_expiry"))
#         self.df = self.df.join(closest_expiry,on='UTC-Datetime')
#         self.df = self.df.filter(pl.col("contract_expiry")==pl.col("min_expiry"))
#         self.df = self.df.drop(['month_code','contract_year','contract_month','contract_expiry','expiry','min_expiry'])
        
        return self


    def compute_remain_ttm(self):
        
        """
        This method calculates remaining maturity until the near month and far month in days and years
        This brings in the expiration dates using expiration_calendar module (which now taken into account expiration "time")
        Previously calculated time to maturity as EOD but now, its time to maturity is calculated until 9:30AM ET
        
        """
      
        # Step 1: Fully lazy min/max extraction, pulls out minimum and maximum date as expression
        minmax_expr = self.df.select([
            pl.col("UTC-Datetime").min().alias("start_date"),
            pl.col("UTC-Datetime").max().alias("end_date")
        ])
        minmax = minmax_expr.collect()
        start_date = minmax[0, "start_date"]
        end_date = minmax[0, "end_date"]

        # Step 2: Create 1min interval datetime range (pl.duration doesnt support minute interval) - eagerframe
        minute_df = pl.DataFrame({
            "UTC-Datetime": pl.datetime_range(
                start=start_date,
                end=end_date ,
                interval='1m',
                time_unit="ns",
                eager=True
            )
        })

        # Step 3: Lazy join back to original df via asof
        # By using join_asof using strategy= 'backward', the sofr data is forward filled
        self.df = minute_df.lazy().join_asof(self.df, left_on="UTC-Datetime", right_on="UTC-Datetime", strategy="backward")

        # Step 4: Expiration table (eager since expiration_list comes externally)
        # The way how we find nearest and far expiry is by using join_asof. 
        # By using strategy='forward' we look for nearest value forward in time
        expiration = self.calendar.get_expiration(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        df_near = pl.DataFrame({"NearExpiry": expiration[:-1], "Date_Near": expiration[:-1]
                                }).with_columns([
                                pl.col("NearExpiry").cast(pl.Datetime("ns", "UTC")),
                                pl.col("Date_Near").cast(pl.Datetime("ns", "UTC"))
                            ]).lazy()
        df_far = pl.DataFrame({"FarExpiry": expiration[1:], "Date_Near": expiration[:-1]
                               }).with_columns([
                                pl.col("FarExpiry").cast(pl.Datetime("ns", "UTC")),
                                pl.col("Date_Near").cast(pl.Datetime("ns", "UTC"))
                            ]).lazy()

        # Step 5: Join near/far expiry onto main dataframe
        self.df = self.df.join_asof(df_near, left_on="UTC-Datetime", right_on="Date_Near", strategy="forward")
        self.df = self.df.join(df_far, on="Date_Near", how="left")

        self.df = self.df.with_columns([
        pl.col("NearExpiry").cast(pl.Datetime("ns", "UTC")),
        pl.col("FarExpiry").cast(pl.Datetime("ns", "UTC"))
    ])

        # Step 6: Compute days and years to each expiry (fully lazy)
        # This calculation needed because we want exact maturity time
        days_in_seconds = 60 * 60 * 24
        years_in_seconds = days_in_seconds * 365

        self.df = self.df.with_columns([
            ((pl.col("NearExpiry") - pl.col("UTC-Datetime")).dt.total_seconds() / days_in_seconds).alias("NearMonth_Days"),
            ((pl.col("FarExpiry") - pl.col("UTC-Datetime")).dt.total_seconds() / days_in_seconds).alias("FarMonth_Days"),
            ((pl.col("NearExpiry") - pl.col("UTC-Datetime")).dt.total_seconds() / years_in_seconds).alias("NearMonth_Years"),
            ((pl.col("FarExpiry") - pl.col("UTC-Datetime")).dt.total_seconds() / years_in_seconds).alias("FarMonth_Years")
        ])
        
        

        return self
    

    def compute_sofr_years(self):
        """
        This method converts Term SOFR month to years
        This now handles different month days (e.g March 31days, April : 30days)
        """

        # This should be changed to 1M , 3M , 6M , 12M , once we have the data
        sofr_columns = ["TSFR1M Index", "TSFR3M Index", "TSFR6M Index", "TSFR12M Index"]

        index_columns = [
            "UTC-Datetime",
            "NearMonth_Days",
            "FarMonth_Days",
            "NearMonth_Years",
            "FarMonth_Years",
        ]
        self.df = (
            self.df.unpivot(index=index_columns, on=sofr_columns)
            .rename({"variable": "Tenor", "value": "Rate"})
            .sort(["UTC-Datetime"])
        )
        
        self.df = self.parse_dates().get()
        
        convert_rate = 365/360
         # old code:
        # self.df = self.df.with_columns(
        #     [
        #         pl.when(pl.col("Rate").is_not_null())
        #         .then(pl.col("Rate") * convert_rate)
        #         .otherwise(None)
        #     ]
        # )

        # updated code to convert simple rate into continuously compounded rate:
        # r_simple_pct   = Rate * convert_rate
        # r_simple_dec   = r_simple_pct / 100
        # total_cc       = ln(1 + r_simple_dec * Reference_Year)
        # annual_cc      = total_cc / Reference_Year
        # r_cc_pct       = annual_cc * 100
        self.df = self.df.with_columns(
                pl.when(pl.col("Rate").is_not_null())
                .then(((1 + (pl.col("Rate") * convert_rate) / 100 * pl.col("Reference_Year"))
                .log() / pl.col("Reference_Year")) * 100)
                .otherwise(None)
                .alias("Rate")
        )
        # restore a bare Date column for any later unique()/sort()/join on "Date"
        self.df = self.df.with_columns(pl.col("UTC-Datetime").alias("Date"))

        
        return self

    def parse_dates(self):
        """
        Adds Reference_Date and Reference_Year columns by parsing tenor from 'Tenor' column
        and shifting 'Date' column by the appropriate amount using relativedelta.
        This parses month part of data and then finds the right month days.
        """
        

        def shift_by_tenor(date, tenor):
            # TSFR1M Index -> 1M
            # num = 1, dt = M
            tenor_clean = tenor.replace("TSFR", "").replace("Index", "").strip()
            num = ''.join(filter(str.isdigit, tenor_clean))
            dt = ''.join(filter(str.isalpha, tenor_clean))

            if dt == "M":
                return date + relativedelta(months=int(num))
            else:
                raise ValueError(f"Unsupported tenor type: {tenor}")

        # Step 1: extract Date and Tenor eagerly
        df_eager = self.df.select(["UTC-Datetime", "Tenor"]).collect()

        # Step 2: apply Python shift
        reference_dates = [
            shift_by_tenor(date, tenor)
            for date, tenor in zip(df_eager["UTC-Datetime"], df_eager["Tenor"])
        ]

        # Step 3: make a new column and add to LazyFrame

        df_ref = pl.DataFrame({"Reference_Date": reference_dates}).with_columns(
            pl.col("Reference_Date").cast(pl.Datetime("ns", "UTC"))
        )

        # Step 4: concatenate horizontally
        self.df = self.df.with_columns(df_ref)

        # Step 5: compute Reference_Days and Reference_Year
        self.df = self.df.with_columns([
            (pl.col("Reference_Date") - pl.col("UTC-Datetime")).dt.total_days().alias("Reference_Days"),
            ((pl.col("Reference_Date") - pl.col("UTC-Datetime")).dt.total_days() / 365.0).alias("Reference_Year")
        ])
        
        return self

    def align_sofr_curve(self):
        """align rates and take care of null values
        remove rows with Rates having null values
        We need this process so that we are not putting null values in the interpolator
        Also, Lazyframe doesnt guarantee the order of dataframe so we need sorting
        """

        self.df = self.df.filter(pl.col("Rate").is_not_null())
        self.df = self.df.sort(by=["UTC-Datetime", "Reference_Date"], descending=[False, False])

        return self

    def build_term_structure(self):
        """
        This produces near month and far month interest rate corresponding to each tenor
        by running interpolation method we set.


        """
        df = (
            self.df.collect().to_pandas()
        )  # LazyFrame → eager DataFrame → pandas conversion

        results = []
        for date, group in df.groupby("UTC-Datetime"):
            x = group["Reference_Year"].to_numpy()
            y = group["Rate"].to_numpy()

            near = group["NearMonth_Years"].to_numpy()[0]
            far = group["FarMonth_Years"].to_numpy()[0]

            self.build_interest_rate_model(params={"times": x, "values": y})
            near_rate = self.interest_rate_model.get_rate(near)
            far_rate = self.interest_rate_model.get_rate(far)

            results.append({"UTC-Datetime": date.isoformat(), "Near_Rate": near_rate, "Far_Rate": far_rate})

        result_df = pl.DataFrame(results).with_columns(
    [
        pl.col("UTC-Datetime").str.strptime(pl.Datetime("ns", time_zone="UTC"), strict=False),
        pl.col("Near_Rate").cast(pl.Float64),
        pl.col("Far_Rate").cast(pl.Float64),
    ]
    ).lazy()

        # 
        self.df = self.df.join(result_df, on="UTC-Datetime", how="left")
        
        return self
        
    
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
            
        # First converts to your preset timezone and then converts back to UTC
        utc_datetime_expr = (
        pl.datetime(
            year=pl.col("Date").dt.year(),
            month=pl.col("Date").dt.month(),
            day=pl.col("Date").dt.day(),
            hour=pl.lit(self.batch_time), 
            time_unit="ns"
        )
        .dt.replace_time_zone(self.timezone)
        .dt.convert_time_zone("UTC")
    )

        self.df = self.df.with_columns(
        utc_datetime_expr.alias("UTC-Datetime")
    )

        return self


    def timezone_str_convert(self):
        """
        This converts Prime Trading provided dataframe column Date-Time into timezone aware datetime instance
        """

        # Parses string to datetime instance and then to UTC 
        self.df = self.df.with_columns(
            (
                pl.col("Date-Time").str.strptime(
                    pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%9fZ", strict=True
                )
            )
            .dt.replace_time_zone("UTC")
            .alias("Date-Time")
        ).rename({"Date-Time":"UTC-Datetime"})
        
        # Use GMT Offset information to add CDT column Date
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
        """
        This is a pipeline for sofr data
        """
        return (
            self.check_data_type().add_time_info() 
            .compute_remain_ttm() 
            .compute_sofr_years()
            .align_sofr_curve()
            .build_term_structure() #.convert_to_continuous_compounding_rates()
            # .convert_to_utc()
            .get()
        )

    def run_all2(self):
        # for dividend index
        return self.add_time_info().convert_to_utc().get()

    def get(self) -> pl.DataFrame:
        return self.df


if __name__ == "__main__":
    MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
    calendar = ExpirationCalendar(contract_type="es")
    
    es_trade_path = Path(MANUAL_DATA_DIR / "ES_1yr_Trade.csv.gz")
    es_trade = Preprocessor(dir=es_trade_path, calendar=calendar)
    
    
    
    
