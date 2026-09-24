import jax
import jax.numpy as jnp

import pymanopt


key = jax.random.key(42)

dim = 3
manifold = pymanopt.manifolds.Sphere(dim)

matrix = jax.random.normal(key, shape=[dim, dim])
matrix = 0.5 * (matrix + matrix.T)


@pymanopt.function.jax(manifold)
def cost(point):
    return -point @ matrix @ point


problem = pymanopt.Problem(manifold, cost)

optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem)

eigenvalues, eigenvectors = map(jnp.real, jnp.linalg.eig(matrix))
dominant_eigenvector = eigenvectors[:, eigenvalues.argmax()]

print("Dominant eigenvector:", dominant_eigenvector)
print("Pymanopt solution:", result.point)
