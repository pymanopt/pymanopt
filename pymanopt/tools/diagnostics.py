import numpy as np
from numpy import linalg as la

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def identify_linear_piece(x, y, window_length):
    """Identify a segment of the curve (x, y) that appears to be linear.
    This function attempts to identify a contiguous segment of the curve
    defined by the vectors x and y that appears to be linear. A line is fit
    through the data over all windows of length window_length and the best
    fit is retained. The output specifies the range of indices such that
    x(segment) is the portion over which (x, y) is the most linear and the
    output poly specifies a first order polynomial that best fits (x, y) over
    that segment, following the usual matlab convention for polynomials
    (highest degree coefficients first).
    See also: check_directional_derivative check_gradient check_hessian
    """
    residues = np.zeros(len(x)-window_length)
    polys = np.zeros(shape=(2, len(residues)))
    for i in range(len(residues)):
        segment = range(i, (i+window_length)+1)
        poly, residuals, _, _, _ = np.polyfit(x[segment], y[segment],
                                              1, full=True)
        residues[i] = la.norm(residuals)
        polys[:, i] = poly
    best = np.argmin(residues)
    segment = range(best, best+window_length+1)
    poly = polys[:, best]
    return segment, poly


def get_directional_derivative(problem, x, d):
    """Computes the directional derivative of the cost function at x along d.
    Returns the derivative at x along d of the cost function described in the
    problem structure.
    """
    if hasattr(problem.manifold, "diff"):
        diff = problem.manifold.diff(x, d)
    else:
        grad = problem.manifold.grad(x)
        diff = problem.manifold.inner(x, grad, d)
    return diff


def check_directional_derivative(problem, x=None, d=None,
                                 force_gradient=False):
    """Checks the consistency of the cost function and directional derivatives.
    check_directional_derivative performs a numerical test to check that the
    directional derivatives defined in the problem structure agree up to first
    order with the cost function at some point x, along some direction d. The
    test is based on a truncated Taylor series (see online pymanopt
    documentation).
    Both x and d are optional and will be sampled at random if omitted.
    See also: check_gradient check_hessian
    If force_gradient is True, then the function will call get_gradient and
    infer the directional derivative, rather than call
    get_directional_derivative directly. This is used by check_gradient.
    """
    #  If x and / or d are not specified, pick them at random.
    if d is not None and x is None:
        raise ValueError("If d is provided, x must be too, "
                         "since d is tangent at x.")
    if x is None:
        x = problem.manifold.rand()
    elif x.shape != problem.manifold.rand().shape:
        x = np.reshape(x, problem.manifold.rand().shape)
    if d is None:
        d = problem.manifold.randvec(x)
    elif d.shape != problem.manifold.randvec(x).shape:
        d = np.reshape(d, problem.manifold.randvec(x).shape)

    # Compute the value f0 at f and directional derivative at x along d.
    f0 = problem.cost(x)
    if not force_gradient:
        df0 = get_directional_derivative(problem, x, d)
        pass
    else:
        grad = problem.grad(x)
        df0 = problem.manifold.inner(x, grad, d)

    #  Pick a stepping function: exponential or retraction?
    if hasattr(problem.manifold, "exp"):
        stepper = problem.manifold.exp
    else:
        # No need to issue a warning: to check the gradient, any retraction
        # (which is first-order by definition) is appropriate.
        stepper = problem.manifold.retr

    # Compute the value of f at points on the geodesic (or approximation
    # of it) originating from x, along direction d, for stepsizes in a
    # large range given by h.
    h = np.logspace(-8, 0, 51)
    value = np.zeros_like(h)
    for i, h_k in enumerate(h):
        y = stepper(x,  h_k * d)
        value[i] = problem.cost(y)

    # Compute the linear approximation of the cost function using f0 and
    # df0 at the same points.
    model = np.polyval([df0, f0], h)

    # Compute the approximation error
    err = np.abs(model - value)

    if not np.all(err < 1e-12):
        is_model_exact = False
        # In a numerically reasonable neighborhood, the error should
        # decrease as the square of the stepsize, i.e., in loglog scale,
        # the error should have a slope of 2.
        window_len = 10
        segment, poly = identify_linear_piece(np.log10(h), np.log10(err),
                                              window_len)
    else:
        is_model_exact = True
        # The 1st order model is exact: all errors are (numerically) zero
        # Fit line from all points, use log scale only in h.
        segment = range(len(h))
        poly = np.polyfit(np.log10(h), err, 1)
        # Set mean error in log scale for plot.
        poly[-1] = np.log10(poly[-1])

    if is_model_exact:
        print("Directional derivative check. "
              "It seems the linear model is exact: "
              "Model error is numerically zero for all h.")
    else:
        print("Directional derivative check. The slope of the "
              "continuous line should match that of the dashed "
              "(reference) line over at least a few orders of "
              "magnitude for h.")
    return h, err, segment, poly


def check_gradient(problem, x=None, d=None):
    """Checks the consistency of the cost function and the gradient.
    check_gradient performs a numerical test to check that the gradient
    defined in the problem structure agrees up to first order with the cost
    function at some point x, along some direction d. The test is based on a
    truncated Taylor series (see online pymanopt documentation).
    It is also tested that the gradient is indeed a tangent vector.
    Both x and d are optional and will be sampled at random if omitted.
    """
    #  If x and / or d are not specified, pick them at random.
    if plt is None:
        raise RuntimeError("The 'check_gradient' function requires matplotlib")
    if d is not None and x is None:
        raise ValueError("If d is provided, x must be too,"
                         "since d is tangent at x.")
    if x is None:
        x = problem.manifold.rand()
    elif x.shape != problem.manifold.rand().shape:
        x = np.reshape(x, problem.manifold.rand().shape)
    if d is None:
        d = problem.manifold.randvec(x)
    elif d.shape != problem.manifold.randvec(x).shape:
        d = np.reshape(d, problem.manifold.randvec(x).shape)

    h, err, segment, poly = check_directional_derivative(problem, x, d,
                                                         force_gradient=True)

    # plot
    plt.figure()
    plt.loglog(h, err)
    plt.xlabel("h")
    plt.ylabel("Approximation error")
    plt.loglog(h[segment], 10**np.polyval(poly, np.log10(h[segment])),
               linewidth=3)
    plt.autoscale(False)
    plt.plot([1e-8, 1e0], [1e-8, 1e8], linestyle="--", color="k")

    plt.title("Gradient check\nThe slope of the continuous line "
              "should match that of the dashed\n(reference) line "
              "over at least a few orders of magnitude for h.")
    plt.show()

    # Try to check that the gradient is a tangent vector
    if hasattr(problem.manifold, "tangent"):
        grad = problem.grad(x)
        projected_grad = problem.manifold.tangent(x, grad)
        residual = grad - projected_grad
        err = problem.manifold.norm(x, residual)
        print("The residual should be 0, or very close. "
              "Residual: {:g}.".format(err))
        print("If it is far from 0, then the gradient "
              "is not in the tangent space.")
    else:
        print("Unfortunately, pymanopt was unable to verify that the gradient "
              "is indeed a tangent vector. Please verify this manually or "
              "implement the 'tangent' function in your manifold structure.")
        grad = problem.grad(x)
        projected_grad = problem.manifold.proj(x, grad)
        residual = grad - projected_grad
        err = problem.manifold.norm(x, residual)
        print("The residual should be 0, or very close. "
              "Residual: {:g}.".format(err))
        print("If it is far from 0, then the gradient "
              "is not in the tangent space.")
