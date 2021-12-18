Manifolds
=========

The rigorous mathematical definition of a manifold is beyond the scope of this
document.
However, if you are unfamiliar with the idea, it is fine just to visualize it
as a smooth subset of `Euclidean space
<https://en.wikipedia.org/wiki/Euclidean_space>`_.
Simple examples include the surface of a sphere or a `torus
<https://en.wikipedia.org/wiki/Torus>`_, or the `Möbius strip
<https://en.wikipedia.org/wiki/Möbius_strip>`_.
For an exact definition and a general background on Riemannian optimization we
refer readers to the monographs [AMS2008]_ and [Bou2020]_ (both of which are
freely available online).
If you need to solve an optimization problem with a search space that is
constrained in some smooth way, then performing optimization on manifolds may
well be the natural approach to take.

The manifolds that we currently support are listed below.
We plan to implement more depending on the needs of users, so if there is a
particular manifold you would like to optimize over, please let us know.
If you wish to implement your own manifold for Pymanopt, you will
need to inherit from the abstract :class:`pymanopt.manifolds.manifold.Manifold`
or :class:`pymanopt.manifolds.manifold.EuclideanEmbeddedSubmanifold` base
class.

Manifold
--------

.. automodule:: pymanopt.manifolds.manifold

Euclidean Space
---------------

.. automodule:: pymanopt.manifolds.euclidean

Sphere Manifold
---------------

.. automodule:: pymanopt.manifolds.sphere

Stiefel Manifold
----------------

.. automodule:: pymanopt.manifolds.stiefel

Grassmann Manifold
------------------

.. automodule:: pymanopt.manifolds.grassmann
.. automodule:: pymanopt.manifolds.complex_grassmann

Complex Circle
--------------

.. automodule:: pymanopt.manifolds.complex_circle

Special Orthogonal Group
------------------------

.. automodule:: pymanopt.manifolds.special_orthogonal_group

Oblique Manifold
----------------

.. automodule:: pymanopt.manifolds.oblique

Manifold of Positive Semidefinite Matrices
------------------------------------------

.. automodule:: pymanopt.manifolds.psd

Fixed-Rank Matrices
-------------------

.. automodule:: pymanopt.manifolds.fixed_rank

Strictly-Positive Vectors
-------------------------

.. automodule:: pymanopt.manifolds.strictly_positive_vectors

Product Manifold
----------------

.. automodule:: pymanopt.manifolds.product
