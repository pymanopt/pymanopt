import numpy as np


def matrixlincomb(x, a1, u1, a2=None, u2=None):
    """
    Given a point x, two tangent vectors u1 and u2 at x, and two real
    coefficients a1 and a2, returns a tangent vector at x representing
    a1 * u1 + a2 * u2, if u1 and u2 are represented as matrices.

    If a2 and u2 are omitted, the returned tangent vector is a1 * u1.

    The input x is unused.
    """
    y = a1 * u1
    if a2 is not None and u2 is not None:
        return y + a2 * u2
    return y

