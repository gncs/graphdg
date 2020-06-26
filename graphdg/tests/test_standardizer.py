from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from graphdg.standardize import ArrayStandardizer


class TestNormalizers(TestCase):
    def setUp(self):
        self.matrix = np.array([
            [0, 1, 2],
            [1, 2, 4],
        ])

    def test_standardize(self):
        standardizer = ArrayStandardizer.from_array(self.matrix)

        assert_allclose(standardizer.mean, np.array([0.5, 1.5, 3]))
        self.assertEqual(standardizer.mean.shape, (3, ))

        assert_allclose(standardizer.std, np.array([0.5, 0.5, 1]))
        self.assertEqual(standardizer.std.shape, (3, ))

        assert_allclose(self.matrix, standardizer.destandardize(standardizer.standardize(self.matrix)))

    def test_standardizer_fail(self):
        standardizer = ArrayStandardizer.from_array(self.matrix)

        t = np.array([[0, 1]])

        with self.assertRaises(ValueError):
            standardizer.standardize(t)

        with self.assertRaises(ValueError):
            standardizer.destandardize(t)
