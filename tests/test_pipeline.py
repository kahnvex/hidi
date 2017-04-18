import unittest

from hidi.pipeline import Pipeline


class Transform1(object):
    def transform(self, x=2, **kwargs):
        return x*2, kwargs


class Transform2(object):
    def transform(self, x, **kwargs):
        return x + 1, kwargs


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline([Transform1(), Transform2()])

    def test_pipeline_returns_data_and_kwargs(self):
        out = self.pipeline.run(progress=False)
        self.assertEqual(len(out), 2)

    def test_pipeline_applies_transformations(self):
        output, kwargs = self.pipeline.run(progress=False)
        self.assertEqual(output, 5)

    def test_pipeline_run_function_takes_initial_input(self):
        output, kwargs = self.pipeline.run(4, progress=False)
        self.assertEqual(output, 9)
