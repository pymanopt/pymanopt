Quickstart
==========

Pymanopt is a modular toolbox and hence easy to use.
All of the automatic differentiation is done behind the scenes so that the
amount of setup the user needs to do is minimal.
Usually only the following steps are required:

#. Instantiate a manifold :math:`\manM` from the :mod:`pymanopt.manifolds`
   package to optimize over.
#. Define a cost function :math:`f:\manM \to \R` to minimize using one of the
   backend decorators defined in :mod:`pymanopt.function`.
   The decorator selects the :ref:`backend <Backends>` used to compute
   derivatives and to carry out computations on the manifold.
#. Create a :class:`pymanopt.Problem` instance tying the
   optimization problem together.
#. Instantiate a Pymanopt optimizer from :mod:`pymanopt.optimizers` and run it
   on the problem instance.

Installation
------------

Pymanopt is compatible with Python 3.11+, and depends on NumPy and SciPy.
Additionally, to use Pymanopt's built-in automatic differentiation, which we
strongly recommend, you need to setup your cost functions using either
`Autograd <https://github.com/HIPS/autograd>`_,
`JAX <https://jax.readthedocs.io/en/latest/>`_,
`TensorFlow <https://www.tensorflow.org>`_ or
`PyTorch <http://www.pytorch.org/>`_.
If you are unfamiliar with these packages and you are unsure which to go for,
we suggest to start with JAX.
Its ``jax.numpy`` module mirrors the NumPy API.
To get the latest version of Pymanopt, install it via `uv` or `pip` by
specifying the backend(s) you want to use among `autograd`, `jax`, `torch` and
`tensorflow`, separated by a comma. For example:

.. code-block:: bash

    $ uv add "pymanopt[jax]" # or [autograd], [tensorflow], [torch]
    $ pip install "pymanopt[jax]"

A Simple Example
----------------

As a simple illustrative example, we consider the problem of estimating the
dominant eigenvector of a real symmetric matrix :math:`\vmA \in \R^{n \times
n}`.
As is well known, a dominant eigenvector of a matrix :math:`\vmA` is any vector
:math:`\opt{\vmx}` that maximizes the Rayleigh quotient

.. math::

    \begin{align*}
        f(\vmx) &= \frac{\inner{\vmx}{\vmA\vmx}}{\inner{\vmx}{\vmx}}
    \end{align*}

with :math:`\inner{\cdot}{\cdot}` denoting the canonical inner product on
:math:`\R^n`.
The value of :math:`f` at :math:`\opt{\vmx}` coincides with the largest
eigenvalue of :math:`\vmA`.
Clearly :math:`f` is scale-invariant since :math:`f(\vmx) = f(\alpha\vmx)` for
any :math:`\alpha \neq 0`.
Hence one may reframe the dominant eigenvector problem as the minimization
problem

.. math::

    \begin{align*}
        \opt{\vmx} = \argmin_{\vmx \in \sphere^{n-1}}\inner{-\vmx}{\vmA\vmx}
    \end{align*}

with :math:`\sphere^{n-1}` denoting the set of all unit-norm vectors in
:math:`\R^n`: the sphere manifold of dimension :math:`n-1`.

The following is a minimal working example of how to solve the above problem
using Pymanopt for a random symmetric matrix.
As indicated in the introduction above, we follow four simple steps: we
instantiate the manifold, create the cost function (using JAX in this
case), define a problem instance which we pass the manifold and the cost
function, and run the minimization problem using one of the available
optimizers.

.. literalinclude:: ../quickstart.py
   :language: python

Running this example will produce (something like) the following:

.. code-block:: none

    Optimizing...
    Iteration    Cost                       Gradient norm
    ---------    -----------------------    --------------
       1         -7.9623438174948125e-01    9.22035384e-01
       2         -8.9460978200548635e-01    6.07673542e-01
       3         -9.1939146433213770e-01    4.68261477e-01
       4         -9.4170235542174829e-01    2.58907971e-01
       5         -9.5103307942992898e-01    4.30016790e-02
       6         -9.5129598476345201e-01    5.60028823e-03
       7         -9.5129996231864211e-01    4.01521308e-03
       8         -9.5130114805642396e-01    2.91435655e-03
       9         -9.5130124471149735e-01    2.80606283e-03
      10         -9.5130159793154145e-01    2.32123307e-03
      11         -9.5130235898054838e-01    1.80562129e-04
      12         -9.5130236331279749e-01    5.19371591e-05
      13         -9.5130236365024634e-01    2.14626561e-05
      14         -9.5130236371306853e-01    6.42471602e-06
      15         -9.5130236371893284e-01    2.26543628e-06
      16         -9.5130236371922272e-01    2.08447427e-06
      17         -9.5130236371983978e-01    1.61128451e-07
    Terminated - min grad norm reached after 17 iterations, 0.19 seconds.

    Dominant eigenvector: [-0.42056295  0.27035338  0.86604613]
    Pymanopt solution: [ 0.42056304 -0.27035329 -0.86604609]


The solution ``result.point`` is a JAX array since points on a manifold are
arrays of the chosen :ref:`backend <Backends>`.

Note that depending on the random seed used in the example,
the direction of the "true" dominant eigenvector and the
solution found by Pymanopt can differ.
Eigenvectors are not unique since every eigenpair :math:`(\lambda, \vmv)` still
satisfies the eigenvalue equation :math:`\vmA \vmv = \lambda \vmv` if
:math:`\vmv` is replaced by :math:`\alpha \vmv` for some :math:`\alpha \in \R
\setminus \set{0}`.
That is, the dominant eigenvector is only unique up to multiplication by a
nonzero constant; the zero vector is trivially considered *not* an eigenvector.

The example above constitutes the conceivably simplest demonstration of
Pymanopt.
For more interesting examples we refer to the `examples
<https://github.com/pymanopt/pymanopt/tree/master/examples>`_ in Pymanopt's
github repository.
Moreover, `this notebook <examples/notebooks/mixture_of_gaussians.ipynb>`_
demonstrates a more involved application of Riemannian optimization using
Pymanopt in the context of inference in Gaussian mixture models.
