import unittest
import numpy as np

from hidi.factorization import SNMFTransform


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        np_data = np.random.rand(100, 1000)
        items = list(range(0, 100))

        self.snmf = SNMFTransform(rank=3)
        self.out = self.snmf.transform(np_data, items=items)

    def test_returns_item_by_item_shape(self):
        embeddings_matrix, _ = self.out
        self.assertEqual(embeddings_matrix.shape, (100, 3))
