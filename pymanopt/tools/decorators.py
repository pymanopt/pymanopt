from .autodiff import AutogradBackend, PyTorchBackend


def _verify_callable(f):
    if not callable(f):
        raise ValueError("{} is not callable".format(f.__name__))


def autograd(f):
    _verify_callable(f)
    f.backend = AutogradBackend()
    return f


def pytorch(f):
    _verify_callable(f)
    f.backend = PyTorchBackend()
    return f
