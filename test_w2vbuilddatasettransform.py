import unittest
from hidi.factorization import W2VBuildDatasetTransform


class TestW2VBuildDatasetTransform(unittest.TestCase):
    def setUp(self):
        self.words = """Vevo is a awesome!
                        Vevo Vevo
                     """
        self.words = self.words.split()
        self.test_transform = W2VBuildDatasetTransform()

    def test_builddatasettransform_returns_right_type(self):
        x = self.test_transform.transform(self.words)
        data, count, dictionary, reverse_dictionary = x
        self.assertEqual(type([1]), type(data))
        self.assertEqual(type([1]), type(count))
        self.assertEqual(type(dict()), type(dictionary))
        self.assertEqual(type(dict()), type(reverse_dictionary))

    def test_builddatasettransform_returns_correct_dictionary(self):
        x = self.test_transform.transform(self.words)
        data, count, dictionary, reverse_dictionary = x
        self.assertEqual(data[0], 1)
        self.assertEqual(reverse_dictionary[data[0]], 'Vevo')
        self.assertEqual(dictionary['Vevo'], 1)
        self.assertEqual(count[1], ('Vevo', 3))
