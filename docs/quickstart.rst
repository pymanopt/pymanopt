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
#. Create a :class:`pymanopt.Problem` instance tying the
   optimization problem together.
#. Instantiate a Pymanopt optimizer from :mod:`pymanopt.optimizers` and run it
   on the problem instance.

Installation
------------

Pymanopt is compatible with Python 3.6+, and depends on NumPy and SciPy.
Additionally, to use Pymanopt's built-in automatic differentiation, which we
strongly recommend, you need to setup your cost functions using either `JAX
<https://jax.readthedocs.io/en/latest/>_`, `TensorFlow
<https://www.tensorflow.org>`_ or `PyTorch <http://www.pytorch.org/>`_. If you
are unfamiliar with these packages and you are unsure which to go for, we
suggest to start with JAX.
JAX wraps thinly around NumPy, and is very simple to use, particularly if
you're already familiar with NumPy.
To get the latest version of Pymanopt, install it via pip:

.. code-block:: bash

    $ pip install pymanopt

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

.. code-block:: python

    import jax.numpy as jnp
    import pymanopt
    import pymanopt.manifolds
    import pymanopt.optimizers

    key = jax.random.PRNGKey(42)

    dim = 3
    manifold = pymanopt.manifolds.Sphere(dim)

    matrix = jnp.random.normal(key, shape=(dim, dim))
    matrix = 0.5 * (matrix + matrix.T)

    @pymanopt.function.jax(manifold)
    def cost(point):
        return -point @ matrix @ point

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent()
    result = optimizer.run(problem)

    eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
    dominant_eigenvector = eigenvectors[:, eigenvalues.argmax()]

    print("Dominant eigenvector:", dominant_eigenvector)
    print("Pymanopt solution:", result.point)

Running this example will produce (something like) the following:

.. code-block:: none

    Optimizing...
    Iteration    Cost                       Gradient norm
    ---------    -----------------------    --------------
       1         +7.3636104836175786e-01    1.78120267e+00
       2         +1.9196509409063550e-01    1.29654224e+00
       3         +3.0638109168006390e-02    6.13254596e-01
       4         -7.7901008505905957e-03    7.25241389e-02
       5         -8.3457028386552494e-03    5.56249650e-02
       6         -8.4593397499944326e-03    4.49748292e-02
       7         -8.6552155330729182e-03    6.23199857e-03
       8         -8.6576998894642276e-03    3.71419432e-03
       9         -8.6581363887458067e-03    3.06238405e-03
      10         -8.6590589920961022e-03    1.74038637e-04
      11         -8.6590618561912091e-03    3.57952773e-05
      12         -8.6590619610269620e-03    1.48676753e-05
      13         -8.6590619819723654e-03    2.99713741e-06
      14         -8.6590619826608199e-03    1.42672552e-06
      15         -8.6590619827597113e-03    1.01736059e-06
      16         -8.6590619827723175e-03    9.52384449e-07
    Terminated - min grad norm reached after 16 iterations, 0.24 seconds.

    Dominant eigenvector: [ 0.73188691 -0.59568032 -0.33091767]
    Pymanopt solution: [-0.73188694  0.59568023  0.33091777]

Note that the direction of the "true" dominant eigenvector and the solution
found by Pymanopt differ.
This is not surprising though since every eigenpair :math:`(\lambda, \vmv)`
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
