from __future__ import print_function
import unittest
import sys

import numpy as np
from pyspark import SparkContext

import ParallelRegression as PR


class ParallelRegressionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(ParallelRegressionTestCase, cls).setUpClass()
        cls.sc = SparkContext(appName="ParallelRegressionTestCase")

    @classmethod
    def tearDownClass(cls):
        super(ParallelRegressionTestCase, cls).tearDownClass()
        cls.sc.stop()

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
        print('local gradient: {}'.format(local_gradient))
        print('estimated gradient: {}'.format(estimated_gradient))
        for actual, expected in zip(local_gradient, estimated_gradient):
            self.assertLess(abs(actual-expected), 0.0001)


    def test_gradient(self):
        lam = 1.0
        beta = np.array([np.sin(t) for t in range(50)])
        data = PR.readData('small.test', self.sc)
        actual_gradient = PR.gradient(data, beta, lam)
        expected_gradient = PR.estimateGrad(
            lambda beta: PR.F(data, beta, lam),
            beta,
            0.00001)
        print(actual_gradient)
        print(expected_gradient)
