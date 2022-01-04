import numpy as np


def multiprod(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorized matrix-matrix multiplication.

    The matrices ``A`` and ``B`` are assumed to be arrays containing ``k``
    matrices, i.e., ``A`` and ``B`` have shape ``(k, m, n)`` and ``(k, n, p)``,
    respectively.
    The function multiplies each matrix in ``A`` with the corresponding matrix
    in ``B`` along the first dimension.
    The resulting array has shape ``(k, m, p)``.

    Args:
        A: The first matrix.
        B: The second matrix.

    Returns:
        The matrix (or more precisely array of matrices) corresponding to
        the matrix product vectorized over the first dimension of ``A`` and
        ``B`` (if ``A.ndim == 2``).
    """
    if A.ndim == 2:
        return A @ B
    return np.einsum("ijk,ikl->ijl", A, B)


def multitransp(A):
    """Vectorized matrix transpose.

    A is assumed to be an array containing M matrices, each of which has
    dimension N x P.
    That is, A is an M x N x P array. Multitransp then returns an array
    containing the M matrix transposes of the matrices in A, each of which will
    be P x N.
    """
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))


def multihconj(A):
    """Vectorized matrix Hermitian conjugate."""
    return np.conjugate(multitransp(A))


def multisym(A):
    """Vectorized matrix symmetrization.

    Given an array ``A`` of matrices (represented as an array of shape ``(k, n,
    n)``), returns a version of ``A`` with each matrix symmetrized, i.e.,
    every matrix ``A[i]`` satisfies ``A[i] == A[i].T``.
    """
    return 0.5 * (A + multitransp(A))


def multiskew(A):
    """Vectorized matrix skew-symmetrization.

    Same as :func:`multisym`, but returns an array where each matrix ``A[i]``
    is skew-symmetric, i.e., the components of ``A`` satisfy ``A[i] ==
    -A[i].T``.
    """
    return 0.5 * (A - multitransp(A))


def multieye(k, n):
    """Array of ``k`` ``n x n`` identity matrices."""
    return np.tile(np.eye(n), (k, 1, 1))


def multilog(A, pos_def=False):
    """Vectorized matrix logarithm."""
    if not pos_def:
        raise NotImplementedError

    w, v = np.linalg.eigh(A)
    w = np.expand_dims(np.log(w), axis=-1)
    return multiprod(v, w * multitransp(v))


def multiexp(A, sym=False):
    """Vectorized matrix exponential."""
    if not sym:
        raise NotImplementedError

    w, v = np.linalg.eigh(A)
    w = np.expand_dims(np.exp(w), axis=-1)
    return multiprod(v, w * multitransp(v))
