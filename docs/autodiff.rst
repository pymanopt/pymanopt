Automatic Differentiation
=========================

Cost functions, gradients and Hessian-vector products (hvps) in Pymanopt must
be defined as Python callables annotated with one of the :ref:`backend
decorators<Backends>` below.
Decorating a callable with a backend decorator will wrap it in an instance of
the :class:`pymanopt.Function` class that provides a backend-agnostic
API to the :class:`pymanopt.core.problem.Problem` class to compute derivatives.

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

New backends can be created by inheriting from the
:class:`pymanopt.backends.backend.Backend` class, and creating a
backend decorator using :func:`pymanopt.function.decorator_factory`.

.. automodule:: pymanopt.function
