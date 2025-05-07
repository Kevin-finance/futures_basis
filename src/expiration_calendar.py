
import exchange_calendars_extensions.core as ecx
ecx.apply_extensions()
import exchange_calendars as ec
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd


class ExpirationCalendar:
    def __init__(self,contract_type:str, exchange='NYSE',contract_forward=2):
        self.contract_type= contract_type.lower()
        self.roll_months = self._get_roll_schedule()
        self.exchange = ec.get_calendar(exchange) # default exchange to NYSE
        self.contract_forward = contract_forward
        

    def _get_roll_schedule(self):
        if self.contract_type in ['es','nq']:
            return [3,6,9,12]
        elif self.contract_type in ['vix']:
            return list(range(1,13))
        else:
            raise ValueError("No such Contract. Check your ticker")

    def get_expiration(self, start_date:str, end_date:str) -> pd.DatetimeIndex:
        if self.roll_months == [3,6,9,12]:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")+relativedelta(months=self.contract_forward*3)
            end_date = datetime.strftime(end_date, format = "%Y-%m-%d")
            expiration_dates= self.exchange.quarterly_expiries.holidays(start=start_date, end = end_date)
            # can be improved
            # convert timezone naive expiration to ET 9:30AM and then converting back to UTC
            
            expiration_dates = (expiration_dates+pd.Timedelta(hours=9,minutes=30)).tz_localize("America/New_York").tz_convert("UTC")
            
        elif self.roll_months == list(range(1,13)):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")+relativedelta(months=self.contract_forward)
            end_date = datetime.strftime(end_date, format = "%Y-%m-%d")
            expiration_dates= self.exchange.quarterly_expiries.holidays(start=start_date, end = end_date)
        
        else:
            raise ValueError("Expiration Frequency not monthly or quarterly")
        
        return expiration_dates 
    
if __name__ =="__main__":
    ec = ExpirationCalendar('es')
    start = "2023-01-01"
    end = "2024-12-31"
    print(ec.get_expiration(start,end))

    

