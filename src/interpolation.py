from abc import ABC,abstractmethod
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator

    
class InterpolationMethod(ABC):
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def evaluate(self, t):
        pass   

class LinearInterpolation(InterpolationMethod):
    def fit(self, x, y):
        self.curve = interp1d(x, y, kind='linear', fill_value="extrapolate")

    def evaluate(self, t):
        return self.curve(t)

class CubicSplineInterpolation(InterpolationMethod): 
    def fit(self, x, y):
        self.curve = CubicSpline(x,y, bc_type = 'not-a-knot')

    def evaluate(self,t):
        return self.curve(t)


class PiecewiseCubicSplineInterpolation(InterpolationMethod):
    def fit(self, x, y):
        self.curve = PchipInterpolator(x,y,axis=0, extrapolate = True)

    def evaluate(self, t):
        return float(self.curve(t))
    
# class NelsonSiegelInterpolation(InterpolationMethod)
#     def fit(self,x,y):
#         # self.curve = NelsonSiegelInterpolation()

#     def evaluate(self,t):
#         return float(self.curve(t))

def get_interpolator(method: str) :
    if method == "linear":
        return LinearInterpolation()
    elif method == "pwc":
        return PiecewiseCubicSplineInterpolation()
    else:
        raise ValueError(f"Unknown interpolation method: {method}")





