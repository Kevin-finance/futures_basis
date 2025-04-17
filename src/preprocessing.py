import polars as pl
from expiration_calendar import ExpirationCalendar
from interpolation_factory import get_interpolator
from settings import config
from pathlib import Path



class Preprocessor:
    def __init__(self, dir, calendar: ExpirationCalendar, interpolator_method: str = 'linear'):
        self.df = self._load_data()
        self.calendar = calendar  # e.g., ExpirationCalendar('es')
        self.interpolator_method = interpolator_method
        self.interpolator = get_interpolator(method=interpolator_method)

    def _load_data(self , format = "parquet"):
        # Instead of reading it right away on memory(eager execution), waits until query (lazyframe)
        if format == "csv":
            self.df = pl.scan_csv(dir)
            
        elif format == "parquet":
            self.df = pl.scan_parquet(dir)

        else:
            raise ValueError("File format not recognized")
        
        return self

    def parse_dates(self):
        # datetime parsing
        return self

    def compute_ttm(self):
        # TTM for expiration
        return self

    def compute_sofr_days(self):
        # annualize the days
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
    calendar = ExpirationCalendar(contract_type='es')
    print(pl.read_parquet(Path(MANUAL_DATA_DIR)/"index_data.parquet"))
    # a = Preprocessor(dir = Path(MANUAL_DATA_DIR)/"index_data.parquet", calendar = calendar)
    # print(a.get().collect())


