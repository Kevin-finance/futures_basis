import exchange_calendars_extensions.core as ecx
import exchange_calendars as ec
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

ecx.apply_extensions()


class ExpirationCalendar:
    def __init__(self, contract_type: str, exchange="NYSE", contract_forward=2):
        """
        This expiration calendar takes in contract_type and exchange, and how many deferred contract to be pulled out for each date.
        For example, if you want "ES" contract and want front and deferred one then put contract_type = "ES", and contract_forward = 2
        Note that each exchange has different market open days and time, so we take that as a parameter as well.
        Most of the time, they align so its often good choice to use NYSE. CME is not supported at the moment.
        
        """
        self.contract_type = contract_type.lower()
        self.roll_months = self._get_roll_schedule()
        self.exchange = ec.get_calendar(exchange)  # default exchange to NYSE
        self.contract_forward = contract_forward

    def _get_roll_schedule(self):
        """
        Some products have expiry in every quarter sometimes its monthly and for some commodities, 
        they offer non-regular time interval expiration.
        Here we only support quarterly contracts for ES, NQ and VIX for monthly contract.
        However, we can extend this further by adding new contracts.
        """
        if self.contract_type in ["es", "nq"]:
            return [3, 6, 9, 12]
        elif self.contract_type in ["vix"]:
            return list(range(1, 13))
        else:
            raise ValueError("No such Contract. Check your ticker")

    def get_expiration(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        If its quarterly contract, it returns expiration date from start_date to end_date.
        This adds timezone information and convert it back to UTC as well.
        
        """
        if self.roll_months == [3, 6, 9, 12]:
            end_date = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(
                months=self.contract_forward * 3
            )
            end_date = datetime.strftime(end_date, format="%Y-%m-%d")
            expiration_dates = self.exchange.quarterly_expiries.holidays(
                start=start_date, end=end_date
            )
            # can be improved
            # convert timezone naive expiration to ET 9:30AM and then converting back to UTC

            expiration_dates = (
                (expiration_dates + pd.Timedelta(hours=9, minutes=30))
                .tz_localize("America/New_York")
                .tz_convert("UTC")
            )

        elif self.roll_months == list(range(1, 13)):
            end_date = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(
                months=self.contract_forward
            )
            end_date = datetime.strftime(end_date, format="%Y-%m-%d")
            expiration_dates = self.exchange.quarterly_expiries.holidays(
                start=start_date, end=end_date
            )

        else:
            raise ValueError("Expiration Frequency not monthly or quarterly")

        return expiration_dates


if __name__ == "__main__":
    ec = ExpirationCalendar("es")
    start = "2023-01-01"
    end = "2024-12-31"
    print(ec.get_expiration(start, end))
