def get_backend(point):
    backend = None

    while isinstance(point, (tuple, list)):
        point = point[0]

    import numpy as np
    if isinstance(point, np.ndarray):
        backend = "numpy"

    try:
        import torch
        if isinstance(point, torch.Tensor):
            backend = "pytorch"
    except ImportError:
        pass

    try:
        import jax.numpy as jnp
        if isinstance(point, jnp.ndarray):
            backend = "jax"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        if isinstance(point, tf.Tensor):
            backend = "tensorflow"
    except ImportError:
        pass

    if backend is None:
        raise ValueError("Unknown backend...")

    return backend


def numpy_to_backend(point, backend):
    # if point is a namedtuple,
    # convert each point
    if hasattr(point, '_fields'):
        return point.__class__(*[
            numpy_to_backend(p, backend) for p in point])

    # if point has a class that inherits from list or tuple,
    # convert each point
    if issubclass(point.__class__, (tuple, list)):
        return point.__class__([
            numpy_to_backend(p, backend) for p in point])

    if point.dtype.kind == 'c':
        dtype = 'complex128'
    else:
        dtype = 'float64'

    if backend == 'numpy':
        import numpy as np
        point = np.array(point, dtype=dtype)
    elif backend == 'pytorch':
        import torch
        point = torch.tensor(point, dtype=getattr(torch, dtype))
    elif backend == 'jax':
        import jax.numpy as jnp
        point = jnp.array(point, dtype=dtype)
    elif backend == 'tensorflow':
        import tensorflow as tf
        point = tf.convert_to_tensor(point, dtype=getattr(tf, dtype))
    else:
        raise ValueError(f"Unknown backend '{backend}'")

    return point


def array_as(point, as_):
    backend = get_backend(as_)
    return numpy_to_backend(point, backend)
