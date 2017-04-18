import unittest

from hidi.pipeline import Pipeline
from hidi import forking


class Transform1(object):
    def transform(self, x=2, **kwargs):
        return x*2, kwargs


class Transform2(object):
    def transform(self, x, **kwargs):
        return x + 1, kwargs


class Transform3(object):
    def transform(self, x, **kwargs):
        return x + 5, kwargs


class TestPipelineForking(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline([Transform1()])
        self.left = Pipeline([Transform2()])
        self.right = Pipeline([Transform3()])

    def test_pipeline_performs_trivial_fork(self):
        self.pipeline.add(
            forking.TrivialForkTransform([self.left, self.right]))
        self.assertEqual(self.pipeline.run(progress=False),
                         ([(5, {}), (9, {})], {}))

    def test_pipeline_performs_thread_fork(self):
        self.pipeline.add(
            forking.ThreadForkTransform([self.left, self.right]))
        self.assertEqual(self.pipeline.run(progress=False),
                         ([(5, {}), (9, {})], {}))

    def test_pipeline_performs_process_fork(self):
        self.pipeline.add(
            forking.ProcessForkTransform([self.left, self.right]))
        self.assertEqual(self.pipeline.run(progress=False),
                         ([(5, {}), (9, {})], {}))
