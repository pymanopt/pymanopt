import numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing
from scipy.linalg import expm, logm

from pymanopt.tools.multi import (multiexp, multieye, multilog, multiprod,
                                  multisym, multitransp)
from ._test import TestCase


class TestMulti(TestCase):
    def setUp(self):
        self.m = 40
        self.n = 50
        self.p = 40
        self.k = 10

    def test_multiprod_singlemat(self):
        # Two random matrices A (m x n) and B (n x p)
        A = rnd.randn(self.m, self.n)
        B = rnd.randn(self.n, self.p)

        # Compare the products.
        np_testing.assert_allclose(A.dot(B), multiprod(A, B))

    def test_multiprod(self):
        # Two random arrays of matrices A (k x m x n) and B (k x n x p)
        A = rnd.randn(self.k, self.m, self.n)
        B = rnd.randn(self.k, self.n, self.p)

        C = np.zeros((self.k, self.m, self.p))
        for i in range(self.k):
            C[i] = A[i].dot(B[i])

        np_testing.assert_allclose(C, multiprod(A, B))

    def test_multitransp_singlemat(self):
        A = rnd.randn(self.m, self.n)
        np_testing.assert_array_equal(A.T, multitransp(A))

    def test_multitransp(self):
        A = rnd.randn(self.k, self.m, self.n)

        C = np.zeros((self.k, self.n, self.m))
        for i in range(self.k):
            C[i] = A[i].T

        np_testing.assert_array_equal(C, multitransp(A))

    def test_multisym(self):
        A = rnd.randn(self.k, self.m, self.m)

        C = np.zeros((self.k, self.m, self.m))
        for i in range(self.k):
            C[i] = .5 * (A[i] + A[i].T)

        np.testing.assert_allclose(C, multisym(A))

    def test_multieye(self):
        A = np.zeros((self.k, self.n, self.n))
        for i in range(self.k):
            A[i] = np.eye(self.n)

        np_testing.assert_allclose(A, multieye(self.k, self.n))

    def test_multilog_singlemat(self):
        a = np.diag(rnd.rand(self.m))
        q, r = la.qr(rnd.randn(self.m, self.m))
        # A is a positive definite matrix
        A = q.dot(a.dot(q.T))
        np_testing.assert_allclose(multilog(A, pos_def=True), logm(A))

    def test_multilog(self):
        A = np.zeros((self.k, self.m, self.m))
        L = np.zeros((self.k, self.m, self.m))
        for i in range(self.k):
            a = np.diag(rnd.rand(self.m))
            q, r = la.qr(rnd.randn(self.m, self.m))
            A[i] = q.dot(a.dot(q.T))
            L[i] = logm(A[i])
        np_testing.assert_allclose(multilog(A, pos_def=True), L)

    def test_multiexp_singlemat(self):
        # A is a positive definite matrix
        A = rnd.randn(self.m, self.m)
        A = A + A.T
        np_testing.assert_allclose(multiexp(A, sym=True), expm(A))

    def test_multiexp(self):
        A = multisym(rnd.randn(self.k, self.m, self.m))
        e = np.zeros((self.k, self.m, self.m))
        for i in range(self.k):
            e[i] = expm(A[i])
        np_testing.assert_allclose(multiexp(A, sym=True), e)
