import functools

from pymanopt.numerics.core import _not_implemented


@functools.singledispatch
@_not_implemented
def comb(_):
    pass
