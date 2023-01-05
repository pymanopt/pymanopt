from examples._tools import ExampleRunner
from pymanopt.manifolds import Positive
from pymanopt.tools.diagnostics import check_retraction


SUPPORTED_BACKENDS = ("numpy",)


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    m = 128
    n = 64
    manifold = Positive(m, n, k=2)
    check_retraction(manifold)


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Check retraction on positive manifold", SUPPORTED_BACKENDS
    )
    runner.run()
