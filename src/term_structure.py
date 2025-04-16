from abc import ABC,abstractmethod
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator

class TermStructureModel:
    def __init__(self,times,values,interpolator):
        self.interpolator = interpolator
        self.interpolator.fit(times,values)

    def get_rate(self,t):
        return self.interpolator.evaluate(t)
    
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

class CubicSplineInterpolation(InterpolationMethod): # 1 func for all data points
    def fit(self, x, y):
        self.curve = CubicSpline(x,y, bc_type = 'not-a-knot')

    def evaluate(self,t):
        return self.curve(t)


class PiecewiseCubicSplineInterpolation(InterpolationMethod):
    def fit(self, x, y):
        self.curve(x,y,axis=0, extrapolate = True)

    def evaluate(self, t):
        return self.curve(t) # float(self.curve(t))






