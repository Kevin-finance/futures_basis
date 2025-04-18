import polars as pl
import pandas as pd
from expiration_calendar import ExpirationCalendar
from interpolation_factory import get_interpolator
from settings import config
from pathlib import Path



class Preprocessor:
    def __init__(self, dir, calendar: ExpirationCalendar, interpolator_method: str = 'linear'):
        self.df = self._load_data(Path(dir))
        self.calendar = calendar  # e.g., ExpirationCalendar('es')
        self.interpolator_method = interpolator_method
        self.interpolator = get_interpolator(method=interpolator_method)

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


    def convert_sofr_to_annual(self):
        # SOFR is noted in 360 convention thus 365/360
        #overrides SOFR
        convert_rate = 365/360
        self.df = self.df.with_columns([
        pl.when(pl.col("Rate").is_not_null()).then(pl.col("Rate")*convert_rate).otherwise(None)
        ])
        return self
    
    def compute_ttm(self):
        # TTM for expiration, SOFR is noted in 360 convention thus 365/360
        # annualize the days

        #melt the dataframe
        annualize_factor = 365
        self.df = self.df.unpivot(index=["Date",'SPX_PX_LAST',"IDX_EST_DVD_YLD"],on= [col for col in self.df.collect_schema().names() if col.endswith("SOFR")] \
                               ,variable_name = "Tenor",value_name="Rate").sort(['Date','Tenor'])
        self.df = self.df.with_columns([
            pl.col("Tenor").map_elements(lambda t : self.parse_dates(t),return_dtype=pl.Int64)\
                .alias("days_to_maturity")])
        self.df = self.df.with_columns([(pl.col("days_to_maturity")/annualize_factor).alias("TTM")])
        
        return self
    

    def align_sofr_curve(self):
        ## align rates and take care of nans
        return self

    def build_term_structure(self):
        
        return self

    def compute_futures_price(self):
        
        return self

    def clean_columns(self):
        # Remove all unncessary columns
        return self

    def run_all(self):
        # Execute All pipleine
        return (
            self.parse_dates()
                .compute_sofr_days()
                .compute_ttm()
                .align_sofr_curve()
                .compute_futures_price()
                .clean_columns()
                .get()
        )

    def get(self) -> pl.DataFrame:
        return self.df

if __name__ == "__main__":
    MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
    path = Path(MANUAL_DATA_DIR/"index_data.parquet")
    path2 = Path(MANUAL_DATA_DIR/"ES_BTIC_1min.csv.gz")
    calendar = ExpirationCalendar(contract_type='es')

    # a= pl.read_parquet(Path(MANUAL_DATA_DIR)/"index_data.parquet")
    a = Preprocessor(dir = path, calendar = calendar)
    print((a.compute_ttm().convert_sofr_to_annual()).get().collect())
    # print(a.get().collect())


