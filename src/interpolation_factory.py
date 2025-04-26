from interpolation_strategy import LinearInterpolation, PiecewiseCubicSplineInterpolation



def get_interpolator(method: str) :
    if method == "linear":
        return LinearInterpolation()
    elif method == "pwc":
        return PiecewiseCubicSplineInterpolation()
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

