from hidi.transform import Transform
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def map_fn(pipeline, io, progress, **kwargs):
    return pipeline.run(io, progress=progress, **kwargs)


class ExecutorFork(Transform):
    def __init__(self, pipelines, progress=False):
        self.pipelines = pipelines
        self.progress = progress

    def executor_fork(self, Executor, io, **kwargs):
        with Executor(len(self.pipelines)) as ex:
            run = partial(map_fn, io=io, progress=self.progress, **kwargs)

            return list(ex.map(run, self.pipelines)), kwargs


class ThreadForkTransform(ExecutorFork):
    """
    Fork a pipeline using :code:`concurrent.futures.ThreadPoolExecutor`
    as a backend for execution.

    This is useful if you have several transforms that perform well
    when running in concurrent threads such as IO heavy or CPU heavy
    tasks that execute outside the Python runtime.

    The forked transform will return a list of Pipeline outputs,
    in the same order as the forked pipelines were given.

    :param pipelines: An array of pipelines to fork execution to.
    :type pipelines: list[hidi.pipeline.Pipeline]

    :param progress: When True, progress of the forked pipelines
        will be logged.
    :type progress: bool
    """
    def transform(self, io, **kwargs):
        return self.executor_fork(ThreadPoolExecutor, io, **kwargs)


class ProcessForkTransform(ExecutorFork):
    """
    Fork a pipeline using :code:`concurrent.futures.ProcessesPoolExecutor`
    as a backend for execution.

    This method is useful if you have several transforms that
    can be executed concurrently and are CPU intensive.

    The forked pipeline will now return a list of pipeline ouputs,
    in the same order as the forked pipelines were given.

    Special care must be taken as each transform must be pickled
    to a new process.

    :param pipelines: An array of pipelines to fork execution to.
    :type pipelines: list[hidi.pipeline.Pipeline]

    :param progress: When True, progress of the forked pipelines
        will be logged.
    :type progress: bool
    """
    def transform(self, io, **kwargs):
        return self.executor_fork(ProcessPoolExecutor, io, **kwargs)


class TrivialForkTransform(Transform):
    """
    Trivial Fork Transform using an ordinary loop.

    :param pipelines: An array of pipelines to fork execution to.
    :type pipelines: list[hidi.pipeline.Pipeline]

    :param progress: When True, progress of the forked pipelines
        will be logged.
    :type progress: bool
    """
    def __init__(self, pipelines, progress=False):
        self.pipelines = pipelines
        self.progress = progress

    def transform(self, io, **kwargs):
        output = []

        for pipeline in self.pipelines:
            output.append(pipeline.run(io, progress=self.progress, **kwargs))

        return output, kwargs
