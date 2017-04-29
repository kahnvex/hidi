import unittest
import pandas as pd
import numpy as np

from hidi.factorization import W2VStringTransform


class TestW2VStringTransform(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(np.random.randint(
            0, 50, size=(1000, 2)),
            columns=['item_id', 'link_id'])
        self.w2vstring = W2VStringTransform()
        self.string, _ = self.w2vstring.transform(self.df)

    def test_w2vstring_returns_string(self):
        self.assertEqual(type("string"), type(self.string))

    def test_w2vstring_returns_1000X3_item(self):
        df = self.df
        word_list = self.string.split()
        total_item = 0
        for index in df.index.unique():
            total_item = len(df.loc[index].link_id)*3 + total_item
        self.assertEqual(total_item, len(word_list))
