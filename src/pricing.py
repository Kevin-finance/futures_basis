from abc import ABC, abstractmethod
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

class FuturesPricer:
    def __init__(self, rate_model: InterestRateModel, dividend_model: DividendYieldModel):
        self.rate_model = rate_model
        self.dividend_model = dividend_model
    

    def price(self, spot: float, T: float) -> float:

        r = self.rate_model.get_rate(T)
        q = self.dividend_model.get_dividend(T)
        return spot * math.exp((r - q) * T)
    


if __name__=="__main__":
    index_df = pd.read_parquet("../data_manual/index_data.parquet")
    futures_df = pd.DataFrame(columns = ['Date','Index','r','q','TTM_1','Futures'] )

    cme = ExpirationCalendar(contract_type='es')
    method = 'linear'
    linear_interpolator = get_interpolator(method = method)
    

    for i in tqdm(range(index_df.shape[0])):
        current_date = index_df.iloc[i]['Date']
        near_expiry = cme.get_expiration(datetime.strftime(current_date,format='%Y-%m-%d'))[0] 

        TTM = (near_expiry - current_date).days / 365
        spot = index_df.iloc[i]["SPX_PX_LAST"]
        q = index_df.iloc[i]["IDX_EST_DVD_YLD"]

        # You need to take account the interest rates here!
        times = [0.25, 0.5, 1.0]   
        values = [0.03, 0.032, 0.035]  

        term_structure = TermStructureModel(times, values, linear_interpolator)
        r = term_structure.get_rate(TTM)

        # I think we can here do setter instead of creating instance everytime
        rate_model = FlatRateModel(r)
        div_model = FlatDividendModel(q)
        pricer = FuturesPricer(rate_model, div_model)

        futures_df.loc[i, 'TTM_1'] = TTM
        futures_df.loc[i, 'r'] = r
        futures_df.loc[i, 'q'] = q
        futures_df.loc[i, 'Futures'] = pricer.price(spot, TTM)

    