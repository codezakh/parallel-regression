import unittest

import numpy as np

import ParallelRegression as PR


class ParallelRegressionTestCase(unittest.TestCase):
    def test_predict(self):
        vec_a = np.ones((50,))
        vec_b = np.ones((50, 1))
        self.assertEqual(PR.predict(vec_a, vec_b), 50)

    def test_local_gradient(self):
        y = 1.0
        x = np.array([np.cos(t) for t in range(5)])
        beta = np.array([np.sin(t) for t in range(5)])
        local_gradient = PR.localGradient(x, y, beta)
        estimated_gradient = PR.estimateGrad(lambda beta: PR.f(x, y, beta), beta,
        0.00001)
        for actual, expected in zip(local_gradient, estimated_gradient):
            self.assertLess(abs(actual-expected), 0.0001)