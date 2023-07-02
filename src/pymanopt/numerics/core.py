import functools


def _not_implemented(function):
    @functools.wraps(function)
    def inner(*arguments):
        raise TypeError(
            f"Function '{function.__name__}' not implemented for arguments of "
            f"type '{type(arguments[0])}'"
        )

    return inner


@functools.singledispatch
@_not_implemented
def abs(_):
    pass


@functools.singledispatch
@_not_implemented
def allclose(*_):
    pass


@functools.singledispatch
@_not_implemented
def exp(_):
    pass


@functools.singledispatch
@_not_implemented
def tensordot(*_):
    pass
