from tqdm import tqdm


class Pipeline(object):
    """
    Pipeline of transforms.

    Sequentially apply a list of transforms. All steps of the pipeline must be
    'transforms', that is, they must implement transform method. The Pipeline
    abstraction is inspired by the SciKit Learn Pipeline abstraction.

    Takes a list of transform instances.
    """

    def __init__(self, transformers):
        self.transformers = transformers
        self.fork_to = None
        self.fork_method = None

    def wrap_loop(self, to_iterate, progress):
        if progress:
            return tqdm(to_iterate, desc='Pipline Progress')

        return to_iterate

    def add(self, transform):
        """
        Add a transform to the pipeline.
        """
        self.transformers.append(transform)

    def run(self, io=None, progress=True, **kwargs):
        """
        Executes the pipeline and returns the final result.

        Takes an optional io parameter that will serve as input to the initial
        transformer.
        """
        for transformer in self.wrap_loop(self.transformers, progress):

            if io is not None:
                io, kwargs = transformer.transform(io, **kwargs)
            else:
                io, kwargs = transformer.transform(**kwargs)

        return io, kwargs
