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

Pymanopt is compatible with Python 3.8+, and depends on NumPy and SciPy.
Additionally, to use Pymanopt's built-in automatic differentiation, which we
strongly recommend, you need to setup your cost functions using either
`Autograd <https://github.com/HIPS/autograd>`_,
`JAX <https://jax.readthedocs.io/en/latest/>_`,
`TensorFlow <https://www.tensorflow.org>`_ or
`PyTorch <http://www.pytorch.org/>`_.
If you are unfamiliar with these packages and you are unsure which to go for,
we suggest to start with Autograd.
Autograd wraps thinly around NumPy, and is very simple to use, particularly if
you're already familiar with NumPy.
To get the latest version of Pymanopt, install it via `pip` by specifying the 
backend(s) you want to use among `autograd`, `jax`, `torch` and `tensorflow`,
separated by a comma. For example:

.. code-block:: bash

    $ pip install "pymanopt[autograd]" # or [jax], [tensorflow], [torch]

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
instantiate the manifold, create the cost function (using Autograd in this
case), define a problem instance which we pass the manifold and the cost
function, and run the minimization problem using one of the available
optimizers.

.. code-block:: python

    import autograd.numpy as anp
    import pymanopt

    anp.random.seed(42)

    dim = 3
    manifold = pymanopt.manifolds.Sphere(dim)

    matrix = anp.random.normal(size=(dim, dim))
    matrix = 0.5 * (matrix + matrix.T)

    @pymanopt.function.autograd(manifold)
    def cost(point):
        return -point @ matrix @ point

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent()
    result = optimizer.run(problem)

    eigenvalues, eigenvectors = anp.linalg.eig(matrix)
    dominant_eigenvector = eigenvectors[:, eigenvalues.argmax()]

    print("Dominant eigenvector:", dominant_eigenvector)
    print("Pymanopt solution:", result.point)

Running this example will produce (something like) the following:

.. code-block:: none

    Optimizing...
    Iteration    Cost                       Gradient norm
    ---------    -----------------------    --------------
       1         +1.1041943339110254e+00    5.65626470e-01
       2         +5.2849633289004561e-01    8.90742722e-01
       3         -8.0741058657312559e-01    2.23937710e+00
       4         -1.2667369971251594e+00    1.59671326e+00
       5         -1.4100298597091836e+00    1.11228845e+00
       6         -1.5219408277812505e+00    2.45507203e-01
       7         -1.5269956262562046e+00    6.81712914e-02
       8         -1.5273114803528709e+00    3.40941735e-02
       9         -1.5273905588875487e+00    1.70222768e-02
      10         -1.5274100956128560e+00    8.61140952e-03
      11         -1.5274154319869837e+00    3.90706914e-03
      12         -1.5274156215853507e+00    3.62943721e-03
      13         -1.5274162595152783e+00    2.47643452e-03
      14         -1.5274168030609154e+00    3.66398414e-04
      15         -1.5274168133149475e+00    1.45210081e-04
      16         -1.5274168150025758e+00    4.96142583e-05
      17         -1.5274168150483476e+00    4.42317042e-05
      18         -1.5274168151841643e+00    2.13915041e-05
      19         -1.5274168152087644e+00    1.36422863e-05
      20         -1.5274168152220804e+00    6.25780214e-06
      21         -1.5274168152229037e+00    5.48381052e-06
      22         -1.5274168152252021e+00    2.16996083e-06
      23         -1.5274168152255774e+00    7.52279600e-07
    Terminated - min grad norm reached after 23 iterations, 0.01 seconds.

    Dominant eigenvector: [-0.78442334 -0.38225031 -0.48843088]
    Pymanopt solution: [0.78442327 0.38225034 0.48843097]

Note that the direction of the "true" dominant eigenvector and the solution
found by Pymanopt differ.
This is not exactly surprising though.
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
