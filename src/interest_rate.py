from abc import ABC, abstractmethod
import interpolation

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
    
def get_interest_rate_model(model_type: str, params: dict) -> InterestRateModel:
    """
    Factory for creating Interest Rate Models.

    Parameters
    ----------
    model_type : str
        "flat" or "term_structure"
    params : dict
        - if flat: {"rate": float}
        - if term_structure: {"times": array, "values": array, "interpolation_method": str}

    Returns
    -------
    InterestRateModel
    """

    if model_type == "flat":
        return FlatRateModel(rate=params["rate"])
    
    elif model_type == "term_structure":
        interpolator = interpolation.get_interpolator(params["interpolation_method"])
        return TermStructureModel(times=params["times"], values=params["values"], interpolator=interpolator)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

