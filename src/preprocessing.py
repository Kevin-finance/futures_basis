import polars as pl
import pandas as pd
from expiration_calendar import ExpirationCalendar
import interpolation_factory 
import interest_rate_factory 
from settings import config
from pathlib import Path
import numpy as np
from dateutil.relativedelta import relativedelta
import interest_rate_strategy

class Preprocessor:
    def __init__(self, dir, calendar: ExpirationCalendar,  model_type: str = 'term_structure', interpolation_method: str = None):
        """
        Expects dataframe as below and saved in self.df

        - Date: datetime[ns]
        - 1M SOFR: float64
        - 3M SOFR: float64
        - 6M SOFR: float64
        - 12M SOFR: float64

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

    def build_interest_rate_model(self, params: dict):
        """
        params:
        - if flat: {"rate": float}
        - if term_structure: {"times": array, "values": array}
        """

        if self.model_type == "flat":
            self.interest_rate_model = interest_rate_factory.get_interest_rate_model(
                model_type="flat",
                params=params
            )
        elif self.model_type == "term_structure":
            if self.interpolation_method is None:
                raise ValueError("Interpolation method must be provided for term structure model.")
            self.interest_rate_model = interest_rate_factory.get_interest_rate_model(
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
            self.df = pl.scan_csv(path)
            
        elif ".parquet" in suffixes:
            self.df = pl.scan_parquet(path)

        else:
            raise ValueError("File format not recognized")
        return self.df
    

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
        # "1D SOFR", "1W SOFR", "2W SOFR", "3W SOFR",
        "1M SOFR", "2M SOFR", "3M SOFR", "4M SOFR", "6M SOFR"]

        index_columns = ['Date','NearMonth_Days','FarMonth_Days','NearMonth_Years','FarMonth_Years']
        self.df = self.df.unpivot(index=index_columns,on= sofr_columns \
                               ).rename({'variable':"Tenor","value":"Rate"}).sort(['Date','Tenor'])
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


    def parse_dates(self,tenor:str) -> int:
        # Assuming Date format e.g 1D SOFR
        # Can be improved to incorporate different # of days for months
        
        parse_str = tenor.split()[0]
        num, dt = "",""
        for string in parse_str:
            if string.isdigit():
                num+=string
            elif string.isalpha():
                dt += string
        num = int(num)
        
        if dt =="D":
            num *= 1
        elif dt =="W":
            num*=7
        elif dt == "M":
            num*=30
        else:
            raise ValueError("Date Type not supported")
        
        return num

    

    def align_sofr_curve(self):
        """align rates and take care of null values
        remove rows with Rates having null values
        We need this process so that we are not putting null values in the interpolator"""

        self.df = self.df.filter(pl.col("Rate").is_not_null())
        self.df = self.df.sort(by =['Date','Reference_Date'],
                               descending=[False,False])


        return self


    # From here review

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




    def compute_futures_price(self):
        
        return self

    # def clean_columns(self):
    #     # Remove all unncessary columns
    #     return self

    def run_all(self):
        # Execute All pipleine
        return (
            self.compute_remain_ttm()
                .compute_sofr_years()
                .align_sofr_curve()
                .build_term_structure()
                .get()
        )

    



    def get(self) -> pl.DataFrame:
        return self.df

if __name__ == "__main__":
    MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
    path = Path(MANUAL_DATA_DIR/"index_data.parquet")
    # path2 = Path(MANUAL_DATA_DIR/"ES_BTIC_1min.csv.gz")
    calendar = ExpirationCalendar(contract_type='es')

    # a= pl.read_parquet(Path(MANUAL_DATA_DIR)/"index_data.parquet")
    a = Preprocessor(dir = path, calendar = calendar, interpolation_method='linear')
    last = a.run_all().collect()
    last_selected = last.select(['Date','Near_Rate','Far_Rate'])
    last_unique = last_selected.unique(subset=['Date'])
    print(last_unique)
    path2 = Path(MANUAL_DATA_DIR/"PJ_Lab_SPX_DIV_Trade_03_10_03_14_1min.csv.gz")
    div_fut = Preprocessor(dir = path2, calendar = calendar)
    div_idx = pd.read_excel("../data_manual/SPXDIV.xls")
    # GMT summer time ? augment dividend index to the frame and just subtract
    # what time is the dividend index published ? 
    



    print(div_fut.get().collect())
    print(div_idx)


