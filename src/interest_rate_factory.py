from interest_rate_strategy import *
from interpolation_factory import *

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
        interpolator = get_interpolator(params["interpolation_method"])
        return TermStructureModel(times=params["times"], values=params["values"], interpolator=interpolator)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


