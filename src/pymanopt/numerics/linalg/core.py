import functools

from pymanopt.numerics.core import _not_implemented


@functools.singledispatch
@_not_implemented
def cholesky(_):
    pass


@functools.singledispatch
@_not_implemented
def det(_):
    pass


@functools.singledispatch
@_not_implemented
def eigh(_):
    pass


@functools.singledispatch
@_not_implemented
def inv(_):
    pass


@functools.singledispatch
@_not_implemented
def norm(_):
    pass


@functools.singledispatch
@_not_implemented
def qr(_):
    pass


@functools.singledispatch
@_not_implemented
def solve(_):
    pass


@functools.singledispatch
@_not_implemented
def svd(_):
    pass


@functools.singledispatch
@_not_implemented
def solve_continuous_lyapunov(_):
    pass


@functools.singledispatch
@_not_implemented
def expm(_):
    pass


@functools.singledispatch
@_not_implemented
def logm(_):
    pass
