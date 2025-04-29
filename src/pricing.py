from abc import ABC, abstractmethod
import pandas as pd 
import polars as pl
from models.dividend_rate_model import DividendYieldModel, FlatDividendModel
from models.interest_rate_strategy import InterestRateModel, FlatRateModel
from exchange_calendars import ExchangeCalendar
from models.interpolation_factory import get_interpolator
from models.interpolation_strategy import TermStructureModel
from expiration_calendar import ExpirationCalendar
import math
from datetime import datetime
from tqdm import tqdm
import time
from settings import config 
# should consider basis change when dividend happens, earnings occur etc.
# https://www.cmegroup.com/articles/2024/equity-index-dividend-futures-a-primer.html






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
    print(index_df.head())
    # futures_df = pd.DataFrame(columns = ['Date','Index','r','q','TTM_1','Futures'] )

    # import pandas as pd
    # from pandas.tseries.offsets import DateOffset, Week, MonthEnd

    
    # index_df['Date'] = pd.to_datetime(index_df['Date'])


    # index_df['1D_days'] = (index_df['Date'] + DateOffset(days=1) - index_df['Date']).dt.days /365
    # index_df['1W_days'] = (index_df['Date'] + Week(1) - index_df['Date']).dt.days /365
    # index_df['2W_days'] = (index_df['Date'] + Week(2) - index_df['Date']).dt.days /365
    # index_df['1M_days'] = (index_df['Date'] + DateOffset(months=1) - index_df['Date']).dt.days /365
    # index_df['2M_days'] = (index_df['Date'] + DateOffset(months=2) - index_df['Date']).dt.days /365
    # index_df['3M_days'] = (index_df['Date'] + DateOffset(months=3) - index_df['Date']).dt.days /365
    # index_df['4M_days'] = (index_df['Date'] + DateOffset(months=4) - index_df['Date']).dt.days /365
    # index_df['6M_days'] = (index_df['Date'] + DateOffset(months=6) - index_df['Date']).dt.days /365


    # #만기 다음날 튀는듯. 1681행

    # cme = ExpirationCalendar(contract_type='es')
    # method = 'pwc'
    # interpolator = get_interpolator(method = method)
    

    # for i in tqdm(range(index_df.shape[0])):
    #     current_date = index_df.iloc[i]['Date']
    #     near_expiry = cme.get_expiration(datetime.strftime(current_date,format='%Y-%m-%d'))[0] 

    #     TTM = (near_expiry - current_date).days / 365 
    #     spot = index_df.iloc[i]["SPX_PX_LAST"]
    #     q = index_df.iloc[i]["IDX_EST_DVD_YLD"]

    #     # You need to take account the interest rates here!
    #     mask_time = index_df.columns.str.endswith("_days")
    #     mask_rate = index_df.columns.str.endswith("SOFR")

    #     times =  index_df.loc[:,mask_time].iloc[i]
    #     values = index_df.loc[:,mask_rate].iloc[i]  

        
    #     times.index = [c.replace('_days', '') for c in times.index]
    #     values.index = [c.replace(' SOFR', '') for c in values.index]

    #     df_term = pd.concat([times, values], axis=1, keys=["days", "rate"]).dropna()

        
    #     if df_term.empty:
    #         continue



        
    #     term_structure = TermStructureModel(df_term.days, df_term.rate, interpolator)
    #     r = term_structure.get_rate(TTM)

    #     # I think we can here do setter instead of creating instance everytime
    #     rate_model = FlatRateModel(r)
    #     div_model = FlatDividendModel(q)
    #     pricer = FuturesPricer(rate_model, div_model)

    #     futures_df.loc[i, 'TTM_1'] = TTM
    #     futures_df.loc[i, 'r'] = r
    #     futures_df.loc[i, 'q'] = q
    #     futures_df.loc[i, 'Futures'] = pricer.price(spot, TTM)

    #     print(futures_df)

        