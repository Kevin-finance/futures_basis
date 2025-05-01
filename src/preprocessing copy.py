import polars as pl
import pandas as pd
from expiration_calendar import ExpirationCalendar
import interpolation
import interest_rate
from settings import config
from pathlib import Path
import numpy as np
from dateutil.relativedelta import relativedelta
from functools import reduce
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz


class Preprocessor:
    def __init__(self, dir, calendar: ExpirationCalendar,  model_type: str = 'term_structure', interpolation_method: str = None,
                 timezone:str = None , batch_time:int = None):
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
        (See add_ttm_columnwise method)

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
                model_type="flat",
                params=params
            )
        elif self.model_type == "term_structure":
            if self.interpolation_method is None:
                raise ValueError("Interpolation method must be provided for term structure model.")
            self.interest_rate_model = interest_rate.get_interest_rate_model(
                model_type="term_structure",
                params={
                    "times": params["times"],
                    "values": params["values"],
                    "interpolation_method": self.interpolation_method
                }
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return self

    def _load_data(self,path:Path):
        # Instead of reading it right away on memory(eager execution), waits until query (lazyframe)
        suffixes = [s.lower() for s in path.suffixes]
        if ".csv" in suffixes:
            self.df = pl.scan_csv(path,try_parse_dates=False)
                                  
        elif ".parquet" in suffixes:
            self.df = pl.scan_parquet(path)

        else:
            raise ValueError("File format not recognized")
        return self.df
    
    def check_data_type(self):
        """
        This converts Date object to datetime and converts #NA to None which is a typical null in bbg data
        """
        self.df = self.df.with_columns(pl.col("Date").str.strptime(pl.Datetime,format="%m/%d/%Y").dt.cast_time_unit("ns"))
        
        self.df = self.df.with_columns(
            [pl.when(pl.col(col)!="#N/A")
             .then(pl.col(col))
             .otherwise(None)
             .cast(pl.Float64)
             .alias(col) for col, dtype in self.df.schema.items()
             if col!='Date' and dtype==pl.Utf8]
        )

        return self


    def compute_remain_ttm(self):
        """
        This method calculates remaining maturity until the near month and far month in days and years
        FOR FUTURES
        """
        # Checks minimum date and maximum date and save it as a variable(datetime ns)
        start_date = self.df.select(pl.col("Date").min()).collect().item().strftime("%Y-%m-%d")
        end_date = self.df.select(pl.col("Date").max()).collect().item().strftime("%Y-%m-%d")

        # Utilize the expiration_calendar.py 
        expirations = self.calendar.get_expiration(start_date, end_date) # returns DateTimeIndex
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
            self.df = self.df.with_columns([
                pl.Series(name="NearMonth_Days", values=near_days),
                pl.Series(name="FarMonth_Days", values=far_days),
                (pl.Series(name="NearMonth_Days", values=near_days) / 365).alias("NearMonth_Years"),
                (pl.Series(name="FarMonth_Days", values=far_days) / 365).alias("FarMonth_Years"),
            ])
        else:
            raise ValueError("Currently contract forward over 2 not supported")
        
        return self
    
    def compute_sofr_years(self):
        """
        Convert Term sofr to years e.g 1D-> 1/365 years
        """
        # This should be changed to 1M , 3M , 6M , 12M , once we have the data
        sofr_columns = [
        "TSFR1M Index", "TSFR3M Index", "TSFR6M Index", "TSFR12M Index"]

        index_columns = ['Date','NearMonth_Days','FarMonth_Days','NearMonth_Years','FarMonth_Years']
        self.df = self.df.unpivot(index=index_columns,on= sofr_columns \
                               ).rename({'variable':"Tenor","value":"Rate"}).sort(['Date'])
        # ensure polars casts Rate as float
        self.df = self.df.with_columns(pl.col("Rate").cast(pl.Float64))
        self.df = self.df.with_columns([
            pl.col("Tenor").map_elements(lambda t : self.parse_dates(t),return_dtype=pl.Int64)\
                .alias("Reference_Date")])
        self.df = self.df.with_columns([
            (pl.col("Reference_Date") /365).alias("Reference_Year")
        ])   
        
        convert_rate = 365/360
        self.df = self.df.with_columns([
        pl.when(pl.col("Rate").is_not_null()).then(pl.col("Rate")*convert_rate).otherwise(None)
        ])
        return self


    def parse_dates(self, tenor: str) -> int:
        # can be improved!
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
        self.df = self.df.sort(by =['Date','Reference_Date'],
                               descending=[False,False])


        return self


    def build_term_structure(self):
        df = self.df.collect().to_pandas()  # LazyFrame → eager DataFrame → pandas conversion

        results = []
        for date, group in df.groupby("Date"):
            x = group["Reference_Year"].to_numpy()
            y = group["Rate"].to_numpy()
            
            near = group["NearMonth_Years"].to_numpy()[0]
            far = group["FarMonth_Years"].to_numpy()[0]
            

            self.build_interest_rate_model(params={'times':x,'values':y})
            near_rate = self.interest_rate_model.get_rate(near)
            far_rate = self.interest_rate_model.get_rate(far)


            results.append({
                "Date": date,
                "Near_Rate": near_rate,
                "Far_Rate": far_rate
            })

        result_df = pl.DataFrame(results).with_columns([
            pl.col("Near_Rate").cast(pl.Float64),
            pl.col("Far_Rate").cast(pl.Float64),
        ])
        result_df = result_df.with_columns(
            pl.col("Date").cast(pl.Datetime("ns"))
        ).lazy()

        self.df = self.df.join(result_df, on="Date", how="left")

        return self

    def convert_to_utc(self):
        """
        For Bloomberg and data and daily data in general, they are not timezone aware
        Thus we need to use the timezone information to convert it into UTC for better comparison
        """
        # First convert into Datetime from date
        # self.df = self.df.with_columns(pl.col("Date").cast(pl.Datetime(time_unit='ns')))

        # Assigns timezone that was put in as input, and then converts it into UTC
        self.df = self.df.with_columns(pl.col("Date")
                                       .dt.replace_time_zone(self.timezone)
                                       .dt.convert_time_zone("UTC")
                                       .alias("UTC-Datetime"))
        return self

    def add_time_info(self):
        """"
        This methods adds time information for daily data so that we are not looking forward
        e.g. on 4/11 data we are adding 05:00AM so that when we perform join_asof we are using present data
        If we add batch time as 
        """

        # For dividend index data
        if not isinstance(self.df.schema["Date"], pl.Datetime):
            self.df = self.df.with_columns(
                pl.col("Date").str.strptime(pl.Datetime, format="%m/%d/%Y")  
            )
        


        self.df = self.df.with_columns(
        (
            pl.datetime(
                year=pl.col("Date").dt.year(),
                month=pl.col("Date").dt.month(),
                day=pl.col("Date").dt.day(),
                hour=self.batch_time  # ex: 5
            )
            .dt.replace_time_zone(self.timezone)
            .alias("Date")
        )
    )
                                       
        return self

    def timezone_aware(self):
        self.df = self.df.with_columns(
            pl.col("Date").cast(pl.Datetime(time_unit="ns"))
        )
        
        self.df = self.df.with_columns(
            pl.col("Date").dt.replace_time_zone("America/Chicago")
        )

        self.df = self.df.with_columns(
            (pl.col("Date") + pl.duration(hours=5)).alias("Date")
        )
        return self
    
    def timezone_str_convert(self):
        # This is for dividend futures data 
        self.df = self.df.with_columns((pl.col("Date-Time").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%9fZ", strict=False)).alias("Date-Time"))
        self.df = self.df.with_columns((pl.col('Date-Time')+pl.duration(hours=pl.col("GMT Offset"))).alias("Local-Time-UTC"))
        self.df = self.df.with_columns(pl.col("Local-Time-UTC").dt.convert_time_zone("America/Chicago").alias("Local-Time-Chicago"))

        return self
    
    def timezone_str_convert2(self):
        # This is for SPX realtime data
        self.df = self.df.with_columns((pl.col("Date").str.strptime(pl.Datetime,format ="%m/%d/%Y %H:%M",strict=False)).alias("Date"))
        self.df = self.df.with_columns((pl.col('Date')+pl.duration(hours=-1)).dt.replace_time_zone("America/Chicago").alias("Date"))
        # self.df = self.df.with_columns(pl.col("Chicago-Time").dt.convert_time_zone("America/Chicago").alias("Local-Time-Chicago"))

        return self


    @staticmethod                                            
    def merge_asof_tables(*dfs, on: str, strategy: str = "backward", tolerance: str = None):
        def join_func(left, right):
            if tolerance:
                return left.join_asof(right, on=on, strategy=strategy, tolerance=tolerance)
            else:
                return left.join_asof(right, on=on, strategy=strategy)
        
        return reduce(join_func, dfs)



    # def compute_futures_price(self):
        
    #     return self


    def run_all(self):
        # pipeline for sofr
        return (
            self.check_data_type()
            .compute_remain_ttm()
            .compute_sofr_years()
            .align_sofr_curve()
            .build_term_structure()
            .add_time_info().convert_to_utc().get()
        )

    def run_all2(self):
        # for dividend index
        return (
            self.add_time_info().convert_to_utc().get()
        )



    def get(self) -> pl.DataFrame:
        return self.df

if __name__ == "__main__":
    MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")

    # According to CME it distributes SOFR at 5AM CT
    sofr_path = Path(MANUAL_DATA_DIR/"SOFR.csv")
    div_path = Path(MANUAL_DATA_DIR/"SPXDIV.csv")

    divfut_path = Path(MANUAL_DATA_DIR/"PJ_Lab_SPX_DIV_Trade_03_10_03_14_1min.csv.gz")
    divfut_quote_path = Path(MANUAL_DATA_DIR/"PJ_Lab_SPX_DIV_Quote_03_10_03_14.csv.gz")

    spx_path = Path(MANUAL_DATA_DIR/"SPX_1min.csv")

    es_trade_path = Path(MANUAL_DATA_DIR/"PJ_Lab_ES_Trade_03_10_03_14_1min.csv.gz")

    btic_trade_path = Path(MANUAL_DATA_DIR/"PJ_Lab_EST_Trade_03_10_03_14_1min.csv")

    calendar = ExpirationCalendar(contract_type='es')

    sofr = Preprocessor(dir = sofr_path, calendar = calendar, interpolation_method='linear'
                        ,timezone="America/Chicago", batch_time=5)

    div = Preprocessor(dir = div_path, calendar = calendar,
                       timezone="America/New_York",batch_time= 19)
    
    divfut = Preprocessor(dir = divfut_path, calendar = calendar)
    divfut_quote = Preprocessor(dir = divfut_quote_path, calendar = calendar)

    spx = Preprocessor(dir = spx_path, calendar = calendar)
    es_trade = Preprocessor(dir = es_trade_path, calendar = calendar)
    btic_trade = Preprocessor(dir = btic_trade_path, calendar = calendar)




#     divfut_quote_copy = divfut_quote.timezone_str_convert().get().collect()
#     div_copy = div.check_data_type().timezone_aware().get().collect()

#     divfut_quote_copy = divfut_quote_copy.with_columns(
#     ((pl.col("Bid Price") + pl.col("Ask Price")) * 0.5).alias("Mid Price"))
    
#     # Join the dataframe with SPXDIV Index
#     join_df = divfut_quote_copy.join_asof(div_copy, left_on = "Local-Time-Chicago", right_on = 'Date', strategy = 'backward')
#     join_df = join_df.with_columns((pl.col("Mid Price")-pl.col("SPXDIV Index")).alias("Expected Points"))
    

#     es_copy = es_trade.timezone_str_convert().get().collect()
#     es_copy = es_copy.rename({"Local-Time-Chicago":"Date"})
    
#     btic_copy = btic_trade.timezone_str_convert().get().collect()
#     btic_copy = btic_copy.rename({"Local-Time-Chicago":"Date"})



    
#     sofr_df = sofr.run_all().collect().unique(subset=["Date"], keep="first").sort('Date')


#     final_div = join_df.filter(pl.col("Expected Points").is_not_null()).select(["Date","Expected Points"])
#     final_rate = sofr_df.select(['Date','Near_Rate','NearMonth_Years'])
#     final_spx = spx.timezone_str_convert2().get().collect().select(['Date','Close'])
#     final_es = es_copy.select(['Date','Last'])
#     final_btic = btic_copy.select(['Date','Last'])

#     print(final_div)
#     print(final_rate)
#     print(final_spx)
#     print(final_es)
#     print(final_btic)

#     merged = Preprocessor.merge_asof_tables(final_spx,final_rate,final_div,final_es,final_btic,on='Date',strategy= "backward")

#     import pytz
#     chicago = pytz.timezone("America/Chicago")

#     start = chicago.localize(datetime(2025, 3, 10))
#     end = chicago.localize(datetime(2025, 3, 14))

#     merged_filtered = merged.filter(
#         (pl.col("Date") >= start) & (pl.col("Date") <= end)
#     )
#     print(merged_filtered)

#     merged_filtered = merged_filtered.with_columns(
#     ((pl.col("Close") - pl.col("Expected Points")) *
#      np.exp((pl.col("Near_Rate") / 100) * pl.col("NearMonth_Years")))
#     .alias("Theoretical Futures Price")
# )
#     merged_filtered = merged_filtered.rename({"Close":"Spot","Expected Points":"Expected Dividend Points",
#                                               "Last":"ES Trade","Last_right":"BTIC"})
#     print(merged_filtered)
#     merged_filtered = merged_filtered.with_columns((pl.col("Theoretical Futures Price")-pl.col("Spot")).alias("Theoretical Basis"))
#     merged_filtered = merged_filtered.with_columns((pl.col("ES Trade")-pl.col("Spot")).alias("Market Basis"))

#     print(merged_filtered)


#     merged_filtered_pd = merged_filtered.to_pandas()
#     merged_filtered_pd["Date"] = pd.to_datetime(merged_filtered_pd["Date"])

    
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=merged_filtered_pd["Date"], y=merged_filtered_pd["BTIC"],
#         mode='lines+markers',
#         name='BTIC'
#     ))

#     fig.add_trace(go.Scatter(
#         x=merged_filtered_pd["Date"], y=merged_filtered_pd["Theoretical Basis"],
#         mode='lines+markers',
#         name='Theoretical Basis'
#     ))

#     fig.add_trace(go.Scatter(
#         x=merged_filtered_pd["Date"], y=merged_filtered_pd["Market Basis"],
#         mode='lines+markers',
#         name='Market Basis'
#     ))

    
#     fig.update_layout(
#         title="BTIC vs Theoretical Basis vs Market Basis",
#         xaxis_title="Time",
#         yaxis_title="Index Points",
#         legend_title="Legend",
#         hovermode='x unified',
#         template='plotly_white',
#         height=600
#     )

#     fig.show()







