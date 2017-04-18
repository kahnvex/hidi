import unittest
import numpy as np

from hidi.matrix import SVDTransform


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        np_data = np.array([
            [2, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 2, 0, 2, 0],
            [1, 0, 0, 0, 0, 2]
        ])
        items = ['1', '2', '4', '3', '5', '6']

        self.svd = SVDTransform(n_components=3, n_iter=5)
        self.out = self.svd.transform(np_data, items=items)

    def test_returns_item_by_item_shape(self):
        embeddings_matrix, _ = self.out
        self.assertEqual(embeddings_matrix.shape, (6, 3))
