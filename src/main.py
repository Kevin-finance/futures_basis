import pandas as pd 
import polars as pl
from dividend_rate_model import DividendYieldModel, FlatDividendModel
from interest_rate_model import InterestRateModel, FlatRateModel
from exchange_calendars import ExchangeCalendar
from interpolation_factory import get_interpolator
from term_structure import TermStructureModel
from expiration_calendar import ExpirationCalendar
import math
from datetime import datetime
from tqdm import tqdm
import time
from settings import config 

# Data I pulled
# - Index Est. Dividend yield
# - SOFR Curve for interest rate (SOFR is annualized based on 360 where equitiy index should be 365)
# - SPX Index PX_LAST

# To do
# 1. Research and work on interest rate interpolation ( monotonic spline (GPT suggested this haha), linear interpolation etc.)
# 2. Model how futures price should change according to ex-dividend etc.
# 3. Research and model compounding, discrete or continouous ? 
# 4. Data Preprocessing and scalability
# Also consider borrowing rate should be deducted from the fut price
# -> Although modelling is daily we might want to code such that we can incorporate and compare tick data?








MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")



if __name__== "__main__":
    print(MANUAL_DATA_DIR)