Upgrading from 2.x
==================

In Pymanopt 3.0, :ref:`backends <Backends>` provide the arithmetic of manifolds
in addition to derivatives, so manifolds compute with the backend's arrays
instead of NumPy arrays.
The backend decorators and the APIs of :class:`pymanopt.core.problem.Problem`
and the optimizers are unchanged.

#. **The** ``pymanopt.autodiff`` **package was removed.**
   The ``Function`` class moved to :mod:`pymanopt.function`, and backends to
   ``pymanopt.backends``.

#. **Python.**
   Pymanopt requires Python 3.11 or later.

#. **Results are backend arrays.**
   Optimizers no longer convert points to NumPy arrays, e.g. ``result.point``
   is a ``torch.Tensor`` with PyTorch.

   .. code-block:: python

       point = result.point  # 2.x
       point = result.point.numpy()  # 3.0

#. **Inputs are backend arrays.**
   An ``initial_point`` passed to an optimizer must be an array of the backend,
   and so should constants used in the cost function.

#. **Manifolds used outside of a problem need a backend.**
   This affects, among others, ``random_point`` and
   :func:`pymanopt.tools.diagnostics.check_retraction`.

   .. code-block:: python

       manifold = Sphere(3)  # 2.x
       manifold = Sphere(3, backend=NumpyBackend())  # 3.0

#. **Complex manifolds expect a complex data type.**
   Otherwise a warning is emitted.

   .. code-block:: python

       @pymanopt.function.jax(manifold, dtype=JaxBackend.DEFAULT_COMPLEX_DTYPE())
       def cost(point): ...

#. **Custom manifolds.**
   Replace NumPy calls by calls to ``self.backend``, e.g. ``np.cos`` by
   ``self.backend.cos``, and accept a ``backend`` argument that is forwarded to
   the base class constructor.

#. **Custom backends.**
   Backends now implement the interface of
   :class:`pymanopt.backends.backend.Backend`, which adds data type handling
   and the numerical functions used by manifolds.
   The ``is_available`` and ``prepare_function`` methods were removed.
   ``backend_decorator_factory(backend_cls)`` was replaced by
   :func:`pymanopt.function.decorator_factory`, which takes the module and
   class names of the backend.
