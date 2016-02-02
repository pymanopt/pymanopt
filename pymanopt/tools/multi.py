""" Operations on multiple matrices."""
import numpy as np


def multiprod(A, B):
    # Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    # assumed to be arrays containing M matrices, that is, A and B have
    # dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    # in A with the corresponding matrix in B, using matrix multiplication.
    # so multiprod(A, B) has dimensions (M, N, Q).

    # First check if we have been given just one matrix
    if len(np.shape(A)) == 2:
        return np.dot(A, B)

    a = A.reshape(np.hstack([np.shape(A), [1]]))
    b = B.reshape(np.hstack([[np.shape(B)[0]], [1], np.shape(B)[1:]]))
    return np.sum(a * b, axis=2)


def multitransp(A):
    # Inspired by MATLAB multitransp function by Paolo de Leva. A is assumed to
    # be an array containing M matrices, each of which has dimension N x P.
    # That is, A is an M x N x P array. Multitransp then returns an array
    # containing the M matrix transposes of the matrices in A, each of which
    # will be P x N.

    # First check if we have been given just one matrix
    if len(np.shape(A)) == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))


def multisym(A):
    # Inspired by MATLAB multisym function by Nicholas Boumal.
    return .5 * (A + multitransp(A))
