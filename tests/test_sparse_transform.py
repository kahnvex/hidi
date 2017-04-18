import unittest
import numpy as np
import pandas as pd

from hidi.matrix import SparseTransform


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        scores = pd.Series(np.ones(8), dtype=np.float32)
        np_data = np.array([
            [1, 'a'],
            [2, 'b'],
            [4, 'a'],
            [3, 'c'],
            [3, 'b'],
            [5, 'c'],
            [4, 'c'],
            [1, 'b'],
        ])
        col_labels = ['item_id', 'link_id']
        self.input_df = pd.DataFrame(data=np_data, columns=col_labels)
        self.input_df['score'] = scores
        self.sparse = SparseTransform()
        self.out = self.sparse.transform(self.input_df)

    def test_output_is_sparse_matrix(self):
        self.assertEqual(len(self.out), 2)

    def test_links_are_returned(self):
        _, kwargs = self.out
        self.assertEqual(kwargs.get('links'), ['a', 'b', 'c'])

    def test_items_are_returned(self):
        _, kwargs = self.out
        self.assertEqual(kwargs.get('items'), ['1', '2', '4', '3', '5'])

    def test_shape_of_sparse_matrix(self):
        matrix, _ = self.out
        self.assertEqual(matrix.shape, (3, 5))

    def test_zero_x_zero_entry_of_matrix(self):
        matrix, _ = self.out
        self.assertEqual(matrix[0, 0], 1.0)

    def test_zero_x_one_entry_of_matrix(self):
        matrix, _ = self.out
        self.assertEqual(matrix[0, 1], 0.0)
