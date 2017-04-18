class Transform(object):

    err = 'Tranform implementations must implement a transform function'

    def transform(self, *args):
        raise NotImplemented(self.err)

    def merge_kwargs(self, x, y):
        z = x.copy()
        z.update(y)

        return z
