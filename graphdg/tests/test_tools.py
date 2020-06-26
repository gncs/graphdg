from unittest import TestCase

import numpy as np

from graphdg.tools import get_batches, split_by_edges

data_dicts = [
    {
        'globals': [0.5],
        'nodes': [[0.0], [1.0], [2.0], [3.0], [4.0]],
        'edges': [[7.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]],
        'senders': [0, 3, 1, 2],
        'receivers': [3, 1, 2, 3]
    },
    {
        'globals': [1.5],
        'nodes': [[0.0], [1.0]],
        'edges': [[0.5, 1.5]],
        'senders': [0],
        'receivers': [1]
    },
    {
        'globals': [0.7],
        'nodes': [[0.0]],
        'edges': [],
        'senders': [],
        'receivers': []
    },
]


class TestTools(TestCase):
    def setUp(self):
        self.data = [(1, 2), (3, 4), (5, 6)]

    def test_batches(self):
        iterator = get_batches(range(10), 3, seed=42)
        self.assertListEqual(list(next(iterator)), [8, 1, 5])
        self.assertListEqual(list(next(iterator)), [0, 7, 2])
        self.assertListEqual(list(next(iterator)), [9, 4, 3])
        self.assertListEqual(list(next(iterator)), [6])

    def test_batches_large(self):
        iterator = get_batches(range(3), 3, seed=43)
        self.assertListEqual(list(next(iterator)), [1, 2, 0])

        with self.assertRaises(StopIteration):
            next(iterator)

    def test_empty_batch(self):
        iterator = get_batches([], 3, seed=42)
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_batches_small(self):
        iterator = get_batches([], 3, seed=44)
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_edge_split(self):
        with self.assertRaises(AssertionError):
            split_by_edges(data_dicts, np.ones(shape=(4, 1)))

        splits = split_by_edges(data_dicts, np.ones(shape=(5, 1)))
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[0]), 4)
        self.assertEqual(len(splits[1]), 1)
        self.assertEqual(len(splits[2]), 0)
