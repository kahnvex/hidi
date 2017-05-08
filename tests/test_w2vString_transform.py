import unittest
import pandas as pd

from hidi.factorization import W2VStringTransform


class TestW2VStringTransform(unittest.TestCase):
    def setUp(self):
        self.x = ['vevo1', 'vevo2', 'vevo3', 'vevo4',  'vevo5']
        self.y = [1, 1, 1, 2, 3]
        self.df = pd.DataFrame({'link_id': self.y,
                                'item_id': self.x})
        self.w2vstring = W2VStringTransform()
        self.sentences, _ = self.w2vstring.transform(self.df)

    def test_w2vstring_returns_string(self):
        self.assertEqual(type([]), type(self.sentences))

    def test_w2vstring_returns_lenX3_item(self):
        length = sum([len(i) for i in self.sentences])
        self.assertEqual(len(set(self.y)), len(self.sentences))
        self.assertEqual(len(self.x)*3, length)
