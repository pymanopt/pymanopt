import functools

from pymanopt.numerics.core import _not_implemented

# TODO:
#   - np.linalg.cholesky
#   - np.linalg.det
#   - np.linalg.eigh
#   - np.linalg.inv
#   - np.linalg.norm
#   - np.linalg.qr
#   - np.linalg.solve
#   - np.linalg.svd
#   - scipy.linalg.solve_continuous_lyapunov
#   - scipy.linalg.expm
#   - scipy.linalg.logm


@functools.singledispatch
@_not_implemented
def cholesky(_):
    pass
