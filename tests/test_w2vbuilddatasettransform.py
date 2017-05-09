import unittest
from hidi.factorization import W2VBuildDatasetTransform


class TestW2VBuildDatasetTransform(unittest.TestCase):
    def setUp(self):
        self.words = """Vevo is a awesome!
                        Vevo Vevo
                     """
        self.test_transform = W2VBuildDatasetTransform()

    def test_builddatasettransform_returns_right_type(self):
        data, kwargs = self.test_transform.transform(self.words)
        reverse_dictionary = kwargs['reverse_dictionary']
        count = kwargs['item_count']
        self.assertEqual(type([1]), type(data))
        self.assertEqual(type([1]), type(count))
        self.assertEqual(type(dict()), type(reverse_dictionary))

    def test_builddatasettransform_returns_correct_dictionary(self):
        data, kwargs = self.test_transform.transform(self.words)
        reverse_dictionary = kwargs['reverse_dictionary']
        count = kwargs['item_count']
        self.assertEqual(data[0], 1)
        self.assertEqual(reverse_dictionary[data[0]], 'Vevo')
        self.assertEqual(count[1], ('Vevo', 3))
