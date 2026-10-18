"""Regression tests for complete TrustRegions iteration logs."""

import unittest

import numpy as np

from pymanopt import Problem
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds import Euclidean, Sphere
from pymanopt.optimizers import TrustRegions


class TestTrustRegionsLogging(unittest.TestCase):
    def make_problem(self, kind):
        calls = []
        trials = []
        backend = NumpyBackend()
        if kind == "quartic":
            manifold = Euclidean(1, backend=backend)
            initial = np.array([0.1])

            def cost(x):
                return float((x[0] ** 2 - 1) ** 2)

            def gradient(x):
                return 4 * x * (x * x - 1)

            def hessian(x, u):
                return (12 * x * x - 4) * u

            optimum = 0.0
        else:
            diagonal = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
            initial = np.arange(1.0, 6.0)
            if kind == "sphere":
                manifold = Sphere(5, backend=backend)
                initial /= np.linalg.norm(initial)
                optimum = 0.5
            else:
                manifold = Euclidean(5, backend=backend)
                optimum = 0.0

            def cost(x):
                return float(0.5 * x @ (diagonal * x))

            def gradient(x):
                return diagonal * x

            def hessian(x, u):
                return diagonal * u

        def counted_hessian(x, u):
            calls.append(None)
            return hessian(x, u)

        def traced_cost(x):
            trials.append(x.copy())
            return cost(x)

        problem = Problem(
            manifold,
            traced_cost,
            euclidean_gradient=gradient,
            euclidean_hessian=counted_hessian,
        )
        return problem, initial, calls, trials, optimum

    def run_problem(self, kind, use_rand, log_verbosity, limit):
        problem, initial, calls, trials, optimum = self.make_problem(kind)
        counts = [0]

        class ObservedTrustRegions(TrustRegions):
            def _check_stopping_criterion(self, **kwargs):
                counts.append(len(calls))
                return super()._check_stopping_criterion(**kwargs)

        optimizer = ObservedTrustRegions(
            use_rand=use_rand,
            log_verbosity=log_verbosity,
            verbosity=0,
            max_iterations=limit,
            min_gradient_norm=1e-8,
        )
        options = {}
        if kind == "quartic":
            options.update(Delta0=10.0, Delta_bar=10.0)
        # Restore the caller's random state as well as making both runs equal.
        random_state = np.random.get_state()
        try:
            np.random.seed(292)
            result = optimizer.run(problem, initial_point=initial, **options)
        finally:
            np.random.set_state(random_state)
        return result, problem, initial, counts, trials, optimum

    def test_complete_iteration_log(self):
        for kind in ("quadratic", "sphere", "quartic"):
            for use_rand in (False, True):
                for limit in (1, 100):
                    with self.subTest(
                        kind=kind, use_rand=use_rand, limit=limit
                    ):
                        result, problem, initial, counts, trials, optimum = (
                            self.run_problem(kind, use_rand, 1, limit)
                        )
                        off, _, _, off_counts, off_trials, _ = (
                            self.run_problem(kind, use_rand, 0, limit)
                        )
                        # Logging must preserve the numerical trajectory.
                        np.testing.assert_array_equal(trials, off_trials)
                        np.testing.assert_array_equal(result.point, off.point)
                        self.assertEqual(result.cost, off.cost)
                        self.assertEqual(
                            result.gradient_norm, off.gradient_norm
                        )
                        self.assertEqual(result.iterations, off.iterations)
                        self.assertEqual(counts, off_counts)
                        self.assertIsNone(off.log["iterations"])

                        log = result.log["iterations"]
                        self.assertEqual(
                            set(log),
                            {
                                "iteration",
                                "time",
                                "point",
                                "cost",
                                "gradient_norm",
                                "hess_vec_evals",
                            },
                        )
                        for values in log.values():
                            self.assertEqual(
                                len(values), result.iterations + 1
                            )
                        self.assertEqual(
                            log["iteration"],
                            list(range(result.iterations + 1)),
                        )
                        np.testing.assert_array_equal(log["point"][0], initial)
                        np.testing.assert_array_equal(
                            log["point"][-1], result.point
                        )
                        self.assertEqual(log["cost"][-1], result.cost)
                        self.assertEqual(
                            log["gradient_norm"][-1], result.gradient_norm
                        )
                        self.assertEqual(log["hess_vec_evals"][0], 0)
                        self.assertEqual(
                            log["hess_vec_evals"],
                            [0] + np.diff(counts).tolist(),
                        )
                        self.assertEqual(
                            sum(log["hess_vec_evals"]), counts[-1]
                        )
                        self.assertGreater(log["hess_vec_evals"][-1], 0)
                        self.assertTrue(np.all(np.diff(log["time"]) >= 0))
                        self.assertTrue(np.all(np.diff(log["cost"]) <= 1e-12))
                        for point, cost, norm in zip(
                            log["point"], log["cost"], log["gradient_norm"]
                        ):
                            self.assertAlmostEqual(
                                cost, problem.cost(point), places=12
                            )
                            self.assertAlmostEqual(
                                norm,
                                problem.manifold.norm(
                                    point, problem.riemannian_gradient(point)
                                ),
                                places=12,
                            )
                        if limit == 100:
                            self.assertLess(result.gradient_norm, 1e-8)
                            self.assertAlmostEqual(
                                result.cost, optimum, places=10
                            )
                            if kind == "quartic":
                                # Rejected steps still consume Hessian products
                                # and must produce entries, even at the same point.
                                self.assertTrue(
                                    any(
                                        np.array_equal(a, b)
                                        for a, b in zip(
                                            log["point"], log["point"][1:]
                                        )
                                    )
                                )


if __name__ == "__main__":
    unittest.main()
