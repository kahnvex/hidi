import unittest
import numpy as np

from hidi.matrix import SimilarityTransform


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        np_data = np.array([
            [1.0,  0.0,  1.0,  0.0,  0.0],
            [1.0,  1.0,  0.0,  1.0,  0.0],
            [0.0,  0.0,  1.0,  1.0,  1.0]])

        links = ['a', 'b', 'c']
        items = ['1', '2', '4', '3', '5']

        self.sim = SimilarityTransform()
        self.out = self.sim.transform(np_data, links=links, items=items)

    def test_returns_item_by_item_shape(self):
        sim_matrix, _ = self.out
        self.assertEqual(sim_matrix.shape, (5, 5))

    def test_item_is_self_similar(self):
        sim_matrix, _ = self.out
        diagonal = np.diagonal(sim_matrix)
        self.assertEqual(diagonal.tolist(), [2.0, 1.0, 2.0, 2.0, 1.0])

    def test_item_one_vector(self):
        sim_matrix, _ = self.out
        one_vector = sim_matrix[1].tolist()[0]
        self.assertEqual(one_vector, [1.0, 1.0, 0.0, 1.0, 0.0])
