import unittest
import pandas as pd

from hidi.inout import ReadTransform


class TestReadTransform(unittest.TestCase):
    def setUp(self):
        self.data_loader = ReadTransform(
            ['tests/fixtures/test_data_loader_input.csv'])
        self.df, _ = self.data_loader.transform()

    def test_data_loader_returns_dataframe(self):
        self.assertEqual(pd.DataFrame, type(self.df))

    def tests_data_loader_returns_dataframe_with_score(self):
        self.assertEqual(self.df['score'][0], 0.42)

    def test_data_loader_returns_dataframe_with_item_id(self):
        self.assertEqual(self.df['item_id'][0], 74)

    def test_data_loader_returns_dataframe_with_link_id(self):
        self.assertEqual(self.df['link_id'][0], 4)
