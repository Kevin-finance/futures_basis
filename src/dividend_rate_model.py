from abc import ABC, abstractmethod

class DividendYieldModel(ABC):
    @abstractmethod
    def get_dividend(self, t: float) -> float:
        pass

class FlatDividendModel(DividendYieldModel):
    def __init__(self, yield_: float):
        self.yield_ = yield_

    def get_dividend(self, t: float) -> float:
        return self.yield_