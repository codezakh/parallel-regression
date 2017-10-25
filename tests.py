import unittest

import numpy as np

import ParallelRegression as PR


class ParallelRegressionTestCase(unittest.TestCase):
    def test_predict(self):
        vec_a = np.ones((50,))
        vec_b = np.ones((1,50))
        self.assertEqual(PR.predict(vec_a, vec_b), 50)
