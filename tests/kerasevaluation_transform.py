import sys
import unittest
import numpy as np
import pandas as pd

from hidi.matrix import KerasEvaluationTransform

if sys.version_info < (3, 0):
    from mock import Mock
else:
    from unittest.mock import Mock


class TestKeraModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.items = range(10000)
        self.M = pd.DataFrame(np.random.rand(10000, 6), index=self.items)
        self.M.index.name = 'items'
        self.validation = pd.DataFrame(np.random.rand(10000, 2),
                                       index=self.items)
        self.validation.index.name = 'items'

    def test_baselinemodel_fit_is_called_once(self):
        t = KerasEvaluationTransform(Mock, self.validation)
        model, _ = t.transform(self.M)
        self.assertEqual(1, model.fit.call_count)

    def test_baselinemodel_fit_call_args_shape(self):
        t = KerasEvaluationTransform(Mock, self.validation)
        model, _ = t.transform(self.M)
        x_train, y_train = model.fit.call_args[0]
        self.assertEqual(x_train.shape, (7500, 6))
        self.assertEqual(y_train.shape, (7500, 2))