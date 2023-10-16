import functools


def _not_implemented(function):
    @functools.wraps(function)
    def inner(*arguments):
        raise TypeError(
            f"Function '{function.__name__}' not implemented for arguments of "
            f"type '{type(arguments[0])}'"
        )

    return inner


# TODO:
#   - np.finfo
#   - np.hstack
#   - np.iscomplexobj
#   - np.isnan
#   - np.isrealobj
#   - np.log
#   - np.logspace
#   - np.newaxis
#   - np.ones
#   - np.pi
#   - np.polyfit
#   - np.polyval
#   - np.prod
#   - np.random.normal
#   - np.random.randn
#   - np.random.uniform
#   - np.real
#   - np.sin
#   - np.sinc
#   - np.sort
#   - np.spacing
#   - np.sqrt
#   - np.sum
#   - np.tile
#   - np.trace
#   - np.transpose
#   - np.vectorize
#   - np.vstack
#   - np.where
#   - np.zeros
#   - scipy.special.comb


@functools.singledispatch
@_not_implemented
def abs(_):
    pass


@functools.singledispatch
@_not_implemented
def all(_):
    pass


@functools.singledispatch
@_not_implemented
def allclose(*_):
    pass


@functools.singledispatch
@_not_implemented
def any(*_):
    pass


@functools.singledispatch
@_not_implemented
def arange(*_):
    pass


@functools.singledispatch
@_not_implemented
def arccos(_):
    pass


@functools.singledispatch
@_not_implemented
def arccosh(_):
    pass


@functools.singledispatch
@_not_implemented
def arctan(_):
    pass


@functools.singledispatch
@_not_implemented
def arctanh(_):
    pass


@functools.singledispatch
@_not_implemented
def argmin(_):
    pass


@functools.singledispatch
@_not_implemented
def array(_):
    pass


@functools.singledispatch
@_not_implemented
def block(_):
    pass


@functools.singledispatch
@_not_implemented
def conjugate(_):
    pass


@functools.singledispatch
@_not_implemented
def cos(_):
    pass


@functools.singledispatch
@_not_implemented
def diag(_):
    pass


@functools.singledispatch
@_not_implemented
def diagonal(_):
    pass


@functools.singledispatch
@_not_implemented
def eye(_):
    pass


@functools.singledispatch
@_not_implemented
def exp(_):
    pass


@functools.singledispatch
@_not_implemented
def tensordot(*_):
    pass


@functools.singledispatch
@_not_implemented
def tanh(_):
    pass
