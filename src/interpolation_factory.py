from term_structure import LinearInterpolation, PiecewiseCubicSplineInterpolation



def get_interpolator(method: str) :
    if method == "linear":
        return LinearInterpolation()
    elif method == "pwc":
        return PiecewiseCubicSplineInterpolation()

