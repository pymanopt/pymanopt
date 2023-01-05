import numpy as np


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
    that segment (highest degree coefficients first).
    """
    residues = np.zeros(len(x) - window_length)
    polys = np.zeros((2, len(residues)))
    for k in np.arange(len(residues)):
        segment = np.arange(k, k + window_length + 1)
        poly, residuals, *_ = np.polyfit(
            x[segment], y[segment], deg=1, full=True
        )
        residues[k] = np.sqrt(residuals)
        polys[:, k] = poly
    best = np.argmin(residues)
    segment = np.arange(best, best + window_length + 1)
    poly = polys[:, best]
    return segment, poly


def check_directional_derivative(
    problem, x=None, d=None, *, use_quadratic_model=False
):
    """Checks the consistency of the cost function and directional derivatives.

    check_directional_derivative performs a numerical test to check that the
    directional derivatives defined in the problem agree up to first or second
    order with the cost function at some point x, along some direction d. The
    test is based on a truncated Taylor series.
    Both x and d are optional and will be sampled at random if omitted.
    """
    #  If x and / or d are not specified, pick them at random.
    if d is not None and x is None:
        raise ValueError(
            "If d is provided, x must be too, " "since d is tangent at x"
        )
    if x is None:
        x = problem.manifold.random_point()
    if d is None:
        d = problem.manifold.random_tangent_vector(x)

    # Compute the value of f at points on the geodesic (or approximation
    # of it) originating from x, along direction d, for step_sizes in a
    # large range given by h.
    h = np.logspace(-8, 0, 51)
    value = np.zeros_like(h)
    for k, h_k in enumerate(h):
        try:
            y = problem.manifold.exp(x, h_k * d)
        except NotImplementedError:
            y = problem.manifold.retraction(x, h_k * d)
        value[k] = problem.cost(y)

    # Compute the value f0 of f at x and directional derivative at x along d.
    f0 = problem.cost(x)
    grad = problem.riemannian_gradient(x)
    df0 = problem.manifold.inner_product(x, grad, d)

    if use_quadratic_model:
        hessd = problem.riemannian_hessian(x, d)
        d2f0 = problem.manifold.inner_product(x, hessd, d)
        model = np.polyval([0.5 * d2f0, df0, f0], h)
    else:
        model = np.polyval([df0, f0], h)

    # Compute the approximation error
    error = np.abs(model - value)
    model_is_exact = np.all(error < 1e-12)
    if model_is_exact:
        if use_quadratic_model:
            print(
                "Hessian check. "
                "It seems the quadratic model is exact: "
                "model error is numerically zero for all h."
            )
        else:
            print(
                "Directional derivative check. "
                "It seems the linear model is exact: "
                "model error is numerically zero for all h."
            )
        # The model is exact: all errors are (numerically) zero.
        # Fit line from all points, use log scale only in h.
        segment = np.arange(len(h))
        poly = np.polyfit(np.log10(h), error, 1)
        # Set mean error in log scale for plot.
        poly[-1] = np.log10(poly[-1])
    else:
        if use_quadratic_model:
            print(
                "Hessian check. The slope of the "
                "continuous line should match that of the dashed "
                "(reference) line over at least a few orders of "
                "magnitude for h."
            )
        else:
            print(
                "Directional derivative check. The slope of the "
                "continuous line should match that of the dashed "
                "(reference) line over at least a few orders of "
                "magnitude for h."
            )
        window_len = 10
        # Despite not all coordinates of the model being close to the true
        # value, some entries of 'error' can be zero. To avoid numerical issues
        # we add an epsilon here.
        eps = np.finfo(error.dtype).eps
        segment, poly = identify_linear_piece(
            np.log10(h), np.log10(error + eps), window_len
        )
    return h, error, segment, poly


def check_gradient(problem, x=None, d=None):
    """Checks the consistency of the cost function and the gradient.

    check_gradient performs a numerical test to check that the gradient
    defined in the problem agrees up to first order with the cost function at
    some point x, along some direction d. The test is based on a truncated
    Taylor series.

    It is also tested that the gradient is indeed a tangent vector.

    Both x and d are optional and will be sampled at random if omitted.
    """
    if plt is None:
        raise RuntimeError("The 'check_gradient' function requires matplotlib")
    if d is not None and x is None:
        raise ValueError(
            "If d is provided, x must be too, since d must be tangent at x"
        )
    if x is None:
        x = problem.manifold.random_point()
    if d is None:
        d = problem.manifold.random_tangent_vector(x)

    h, error, segment, poly = check_directional_derivative(problem, x, d)

    plt.figure()
    plt.loglog(h, error)
    plt.xlabel("h")
    plt.ylabel("Approximation error")
    plt.loglog(
        h[segment], 10 ** np.polyval(poly, np.log10(h[segment])), linewidth=3
    )
    plt.plot([1e-8, 1e0], [1e-8, 1e8], linestyle="--", color="k")
    plt.title(
        "Gradient check\nThe slope of the continuous line should match that "
        "of the dashed\n(reference) line over at least a few orders of "
        "magnitude for h."
    )
    plt.show()

    grad = problem.riemannian_gradient(x)
    try:
        projected_grad = problem.manifold.to_tangent_space(x, grad)
    except NotImplementedError:
        print(
            "Pymanopt was unable to verify that the gradient is indeed a "
            f"tangent vector since {problem.manifold.__class__.__name__} does "
            "not provide a 'to_tangent_space' implementation."
        )
    else:
        residual = grad - projected_grad
        error = problem.manifold.norm(x, residual)
        print(f"The residual should be close to 0: {error:g}.")
        print(
            "If it is far from 0, then the gradient is not in the tangent "
            "space."
        )


def check_hessian(problem, point=None, tangent_vector=None):
    """Checks the consistency of the cost function and the Hessian.

    The function performs a numerical test to check that the Hessian
    defined in the problem agrees up to second order with the cost function at
    some point x, along some direction d. The test is based on a truncated
    Taylor series.

    It is also tested that the result of applying the Hessian along that
    direction is indeed a tangent vector, and that the Hessian operator is
    linear and symmetric w.r.t. the Riemannian metric.

    Both x and d are optional and will be sampled at random if omitted.
    """
    if plt is None:
        raise RuntimeError("The 'check_hessian' function requires matplotlib")
    if tangent_vector is not None and point is None:
        raise ValueError(
            "If d is provided, x must be too, since d must be tangent at x"
        )
    if point is None:
        point = problem.manifold.random_point()
    if tangent_vector is None:
        tangent_vector = problem.manifold.random_tangent_vector(point)

    h, error, segment, poly = check_directional_derivative(
        problem, point, tangent_vector, use_quadratic_model=True
    )

    plt.figure()
    plt.loglog(h, error)
    plt.xlabel("h")
    plt.ylabel("Approximation error")
    plt.loglog(
        h[segment], 10 ** np.polyval(poly, np.log10(h[segment])), linewidth=3
    )
    plt.plot([1e-8, 1e0], [1e-16, 1e8], linestyle="--", color="k")
    plt.title(
        "Hessian check\nThe slope of the continuous line should match that of "
        "the dashed\n(reference) line over at least a few orders of magnitude "
        "for h."
    )
    plt.show()

    hessian = problem.riemannian_hessian(point, tangent_vector)
    try:
        projected_hessian = problem.manifold.to_tangent_space(point, hessian)
    except NotImplementedError:
        print(
            "Pymanopt was unable to verify that the Hessian operator indeed "
            "yields a tangent vector since "
            f"{problem.manifold.__class__.__name__} does not provide a "
            "'to_tangent_space' implementation."
        )
    else:
        residual = hessian - projected_hessian
        error = problem.manifold.norm(point, residual)
        print(f"The residual should be 0, or very close. Residual: {error:g}.")
        print(
            "If it is far from 0, then the Hessian is not in the tangent "
            "space."
        )
    print()

    # Check linearity of Hessian operator.
    tangent_vector_a = problem.manifold.random_tangent_vector(point)
    tangent_vector_b = problem.manifold.random_tangent_vector(point)
    hessian_a = problem.riemannian_hessian(point, tangent_vector_a)
    hessian_b = problem.riemannian_hessian(point, tangent_vector_b)
    weight_a, weight_b = np.random.normal(size=2)
    hessian_of_linear_combination = problem.riemannian_hessian(
        point, weight_a * tangent_vector_a + weight_b * tangent_vector_b
    )
    linear_combination_of_hessians = (
        weight_a * hessian_a + weight_b * hessian_b
    )
    error_norm = problem.manifold.norm(
        point, hessian_of_linear_combination - linear_combination_of_hessians
    )
    print(
        "The norm of the residual between H[a*d1 + b*d2] and a*H[d1] + "
        f"b*H[d2] should be very close to 0: {error_norm:g}."
    )
    print("If it is far from 0, then the Hessian is not a linear operator.")
    print()

    # Check symmetry of Hessian operator.
    inner_product_a = problem.manifold.inner_product(
        point, tangent_vector_a, hessian_b
    )
    inner_product_b = problem.manifold.inner_product(
        point, hessian_a, tangent_vector_b
    )
    error = inner_product_a - inner_product_b
    print(
        "The difference <d1, H[d2]> - <H[d1], d2> should be close to zero: "
        f"{inner_product_a:g} - {inner_product_b:g} = {error:g}."
    )
    print("If it is far from 0 then the Hessian is not a symmetric operator.")


def check_retraction(manifold, point=None, tangent_vector=None):
    """Check order of agreement between a retraction and the exponential."""
    if plt is None:
        raise RuntimeError(
            "The 'check_retraction' function requires matplotlib"
        )
    if point is None:
        point = manifold.random_point()
        tangent_vector = manifold.random_tangent_vector(point)
    elif tangent_vector is None:
        tangent_vector = manifold.random_tangent_vector(point)

    manifold_class = manifold.__class__.__name__
    try:
        manifold.exp(point, tangent_vector)
    except NotImplementedError:
        raise RuntimeError(
            f"The manifold '{manifold_class}' provides no exponential map as "
            "reference to compare the retraction."
        )
    try:
        manifold.retraction(point, tangent_vector)
    except NotImplementedError:
        raise RuntimeError(
            f"The manifold '{manifold_class}' provides no retraction"
        ) from None
    try:
        manifold.retraction(point, tangent_vector)
    except NotImplementedError:
        raise RuntimeError(
            f"This manifold '{manifold_class}'provides no distance map which "
            "is required to run this check"
        ) from None

    # Compare the retraction and the exponential over steps of varying
    # length, on a wide log-scale.
    step_sizes = np.logspace(-12, 0, 251)
    errors = np.zeros(step_sizes.shape)
    for k, step_size in enumerate(step_sizes):
        errors[k] = manifold.dist(
            manifold.exp(point, step_size * tangent_vector),
            manifold.retraction(point, step_size * tangent_vector),
        )

    # Figure out the slope of the error in log-log, by identifying a piece of
    # the error curve which is mostly linear.
    window_length = 10
    segment, poly = identify_linear_piece(
        np.log10(step_sizes), np.log10(errors), window_length
    )

    print(
        "The slope must be at least 2 to have a proper retraction.\n"
        "For the retraction to be second order, the slope should be 3.\n"
        f"It appears the slope is: {poly[0]}.\n"
        "Note: if the implementation of the exponential map and the\n"
        "retraction are identical, this should be zero: "
        f"{np.linalg.norm(errors)}.\n"
        "In that case, the slope test is irrelevant.",
    )

    plt.figure()
    # Plot the difference between the exponential and the retraction over that
    # span of steps on a doubly-logarithmic scale.
    plt.loglog(step_sizes, errors)
    plt.plot(
        [1e-12, 1e0], [1e-30, 1e6], linestyle="--", color="k", label="Slope 3"
    )
    plt.plot(
        [1e-14, 1e0], [1e-20, 1e8], linestyle=":", color="k", label="Slope 2"
    )
    plt.legend()

    plt.loglog(
        step_sizes[segment],
        10 ** np.polyval(poly, np.log10(step_sizes[segment])),
        linewidth=3,
    )
    plt.xlabel("Step size multiplier t")
    plt.ylabel("Distance between exp(x, v, t) and retraction(x, v, t)")
    plt.title(
        "Retraction check.\nA slope of 2 is required for a valid retraction, "
        "3 is desired."
    )
    plt.show()
