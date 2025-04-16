
import exchange_calendars_extensions.core as ecx
ecx.apply_extensions()
import exchange_calendars as ec
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd


class ExpirationCalendar:
    def __init__(self,contract_type:str, exchange='NYSE'):
        self.contract_type= contract_type.lower()
        self.roll_months = self._get_roll_schedule()
        self.exchange = ec.get_calendar(exchange) # default exchange to NYSE
        

    def _get_roll_schedule(self):
        if self.contract_type in ['es','nq']:
            return [3,6,9,12]
        elif self.contract_type in ['vix']:
            return list(range(1,13))
        else:
            raise ValueError("No such Contract. Check your ticker")

    def get_expiration(self, valuation_date:str) -> pd.DatetimeIndex:
        end_date = datetime.strptime(valuation_date, "%Y-%m-%d")+relativedelta(years=1)
        end_date = datetime.strftime(end_date, format = "%Y-%m-%d")
        if self.roll_months == [3,6,9,12]:
            expiration_dates= self.exchange.quarterly_expiries.holidays(start=valuation_date, end = end_date)
        
        elif self.roll_months == list(range(1,13)):
            expiration_dates= self.exchange.monthly_expiries.holidays(start=valuation_date, end = end_date)            
        
        else:
            raise ValueError("Expiration Frequency not monthly or quarterly")
        
        return expiration_dates[:2] # only extract two nearest expiration dates
    
if __name__ =="__main__":
    ec = ExpirationCalendar('es')
    start = "2025-01-01"
    print(ec.get_expiration(start))
    print(ec.get_expiration(start))



















if __name__ == "__main__":
    print("hi")
