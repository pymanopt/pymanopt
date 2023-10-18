import functools

from pymanopt.numerics.core import _not_implemented


@functools.singledispatch
@_not_implemented
def normal(_):
    pass


@functools.singledispatch
@_not_implemented
def randn(_):
    pass


@functools.singledispatch
@_not_implemented
def uniform(_):
    pass
