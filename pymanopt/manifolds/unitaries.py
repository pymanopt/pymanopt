@staticmethod
def _randskewh(n, N=1):
    # Generate random skew-hermitian matrices with normal entries.
    idxs = np.triu_indices(n, 1)
    S = np.zeros((N, n, n))
    for i in range(N):
        S[i][idxs] = rnd.randn(int(n * (n - 1) / 2))
        S = S - multihconj(S)
    if N == 1:
        return S.reshape(n, n)
    return S


def random_tangent_vector(self, point):
    tangent_vector = self._randskewh(self._n, self._k)
    nrmU = np.sqrt(
        np.tensordot(
            tangent_vector.conj(), tangent_vector, axes=tangent_vector.ndim
        )
    )
    return tangent_vector / nrmU
