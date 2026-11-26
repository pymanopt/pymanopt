Backends
========

Every problem in Pymanopt uses a single backend.
The backend determines both how derivatives of the cost function are computed
and which array type the manifold computes with and returns.

Choosing a Backend
------------------

A backend is selected by decorating the cost function with one of the
decorators from :mod:`pymanopt.function`.

.. list-table::
   :header-rows: 1

   * - Decorator
     - Array type
     - Automatic differentiation
     - ``pip`` extra
   * - :func:`pymanopt.function.numpy`
     - ``numpy.ndarray``
     - no
     - none
   * - :func:`pymanopt.function.autograd`
     - ``numpy.ndarray``
     - yes
     - ``autograd``
   * - :func:`pymanopt.function.jax`
     - ``jax.Array``
     - yes
     - ``jax``
   * - :func:`pymanopt.function.pytorch`
     - ``torch.Tensor``
     - yes
     - ``torch``
   * - :func:`pymanopt.function.tensorflow`
     - ``tf.Tensor``
     - yes
     - ``tensorflow``

With the NumPy backend, derivatives must be passed to
:class:`pymanopt.core.problem.Problem` explicitly, e.g. via its
``euclidean_gradient`` argument.

The signature of the decorated callable must match the point layout of the
manifold it is defined on.
For instance, for memory efficiency points on the
:class:`pymanopt.manifolds.fixed_rank.FixedRankEmbedded` manifold are not
represented as ``m x n`` matrices in the ambient space but rather as singular
value decompositions.
As such a cost function defined on the manifold must accept three arguments
``u``, ``s`` and ``vt``.
Refer to the documentation of the respective manifold on how points are
represented.

How a Manifold Gets Its Backend
-------------------------------

A manifold created without a backend adopts the backend of the cost function
when both are passed to :class:`pymanopt.core.problem.Problem`.

.. code-block:: python

    manifold = pymanopt.manifolds.Sphere(3)

    @pymanopt.function.pytorch(manifold)
    def cost(point):
        return -point @ matrix @ point

    problem = pymanopt.Problem(manifold, cost)  # manifold now uses PyTorch

To use a manifold outside of a problem, e.g. to sample points or to call
:func:`pymanopt.tools.diagnostics.check_retraction`, pass a backend instance
to its constructor.

.. code-block:: python

    from pymanopt.backends.numpy_backend import NumpyBackend

    manifold = pymanopt.manifolds.Sphere(3, backend=NumpyBackend())
    point = manifold.random_point()

If the manifold already has a backend, callables passed to
:class:`pymanopt.core.problem.Problem` need not be decorated.
Decorated ones must use the same backend and precision as the manifold.
The same holds for the manifolds making up a
:class:`pymanopt.manifolds.product.Product` manifold.

Arrays
------

Points and tangent vectors are arrays of the backend's array type.
This includes the ``point`` attribute of the result returned by an optimizer
and an ``initial_point`` passed to it.
Constants used in a cost function should also be arrays of that type, e.g.
``torch.from_numpy(matrix)`` for a NumPy array ``matrix``.

Precision and Complex Manifolds
-------------------------------

Backends use real double precision numbers by default.
The decorators and the backend classes accept a ``dtype`` argument, e.g.
``torch.float32`` for single precision.

Whether a manifold computes with real or complex numbers is determined by the
manifold itself.
If the ``dtype`` of the backend does not match, the manifold switches to the
real or complex data type of the same precision and emits a warning.
To avoid it on complex manifolds such as
:class:`pymanopt.manifolds.grassmann.ComplexGrassmann`, pass a complex
``dtype``.

.. code-block:: python

    from pymanopt.backends.jax_backend import JaxBackend

    @pymanopt.function.jax(manifold, dtype=JaxBackend.DEFAULT_COMPLEX_DTYPE())
    def cost(point):
        ...

Extending Pymanopt
------------------

Custom manifolds should accept a ``backend`` argument, forward it to the
constructor of :class:`pymanopt.manifolds.manifold.Manifold`, and use
``self.backend`` instead of NumPy for all computations, e.g.
``self.backend.linalg_norm(tangent_vector)``.

New backends can be created by inheriting from the
:class:`pymanopt.backends.backend.Backend` class, and creating a backend
decorator using :func:`pymanopt.function.decorator_factory`.

.. code-block:: python

    my_backend = pymanopt.function.decorator_factory(
        "my_package.my_module", "MyBackend"
    )

Function Decorators
-------------------

.. automodule:: pymanopt.function

Backend Interface
-----------------

.. autoclass:: pymanopt.backends.backend.Backend
   :exclude-members: array_t
