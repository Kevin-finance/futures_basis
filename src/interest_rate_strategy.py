from abc import ABC, abstractmethod

class InterestRateModel(ABC):
    @abstractmethod

    def get_rate(self, t: float) -> float:
        pass

class FlatRateModel(InterestRateModel):
    def __init__(self, rate:float):
        self.rate = rate 
    
    def get_rate(self, t: float) -> float:
        return self.rate
    
class TermStructureModel(InterestRateModel):
    def __init__(self,times,values,interpolator):
        self.interpolator = interpolator
        self.interpolator.fit(times,values)

    def get_rate(self,t):
        return self.interpolator.evaluate(t)
