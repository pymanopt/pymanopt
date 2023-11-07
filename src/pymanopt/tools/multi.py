import packaging.version as pv
import scipy.version

import pymanopt.numerics as nx


# Scipy 1.9.0 added support for calling scipy.linalg.expm on stacked matrices.
if pv.parse(scipy.version.version) >= pv.parse("1.9.0"):
    scipy_expm = nx.linalg.expm
else:
    scipy_expm = nx.vectorize(nx.linalg.expm, signature="(m,m)->(m,m)")


def multitransp(A):
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return nx.transpose(A, (0, 2, 1))


def multihconj(A):
    """Vectorized matrix conjugate transpose."""
    return nx.conjugate(multitransp(A))


def multisym(A):
    """Vectorized matrix symmetrization.

    Given an array ``A`` of matrices (represented as an array of shape ``(k, n,
    n)``), returns a version of ``A`` with each matrix symmetrized, i.e.,
    every matrix ``A[i]`` satisfies ``A[i] == A[i].T``.
    """
    return 0.5 * (A + multitransp(A))


def multiherm(A):
    return 0.5 * (A + multihconj(A))


def multiskew(A):
    """Vectorized matrix skew-symmetrization.

    Similar to :func:`multisym`, but returns an array where each matrix
    ``A[i]`` is skew-symmetric, i.e., the components of ``A`` satisfy ``A[i] ==
    -A[i].T``.
    """
    return 0.5 * (A - multitransp(A))


def multiskewh(A):
    return 0.5 * (A - multihconj(A))


def multieye(k, n):
    """Array of ``k`` ``n x n`` identity matrices."""
    return nx.tile(nx.eye(n), (k, 1, 1))


def multilogm(A, *, positive_definite=False):
    """Vectorized matrix logarithm."""
    if not positive_definite:
        return nx.vectorize(scipy.linalg.logm, signature="(m,m)->(m,m)")(A)

    w, v = nx.linalg.eigh(A)
    w = nx.expand_dims(nx.log(w), axis=-1)
    logmA = v @ (w * multihconj(v))
    if nx.isrealobj(A):
        return nx.real(logmA)
    return logmA


def multiexpm(A, *, symmetric=False):
    """Vectorized matrix exponential."""
    if not symmetric:
        return scipy_expm(A)

    w, v = nx.linalg.eigh(A)
    w = nx.expand_dims(nx.exp(w), axis=-1)
    expmA = v @ (w * multihconj(v))
    if nx.isrealobj(A):
        return nx.real(expmA)
    return expmA


def multiqr(A):
    """Vectorized QR decomposition."""
    if A.ndim not in (2, 3):
        raise ValueError("Input must be a matrix or a stacked matrix")

    q, r = nx.vectorize(nx.linalg.qr, signature="(m,n)->(m,k),(k,n)")(A)

    # Compute signs or unit-modulus phase of entries of diagonal of r.
    s = nx.diagonal(r, axis1=-2, axis2=-1).copy()
    s[s == 0] = 1
    s = s / nx.abs(s)

    s = nx.expand_dims(s, axis=-1)
    q = q * multitransp(s)
    r = r * nx.conjugate(s)
    return q, r
