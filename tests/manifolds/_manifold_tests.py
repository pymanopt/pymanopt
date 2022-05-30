import numpy as np
import scipy.stats

import pymanopt
from pymanopt.tools import diagnostics


def run_gradient_test(manifold, cost):
    problem = pymanopt.Problem(manifold, cost)
    h, _, segment, poly = diagnostics.check_directional_derivative(problem)
    # Compute slope of linear regression line through points in linear domain.
    x = np.log(h[segment])
    y = np.log(10) * np.polyval(poly, np.log10(np.e) * x)
    slope = scipy.stats.linregress(x, y).slope
    assert 1.995 <= slope <= 2.005
