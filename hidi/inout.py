import numpy as np
import pandas as pd

from hidi.transform import Transform


class ReadTransform(Transform):
    """
    Read input csv data from disk.

    Input data should be a csv file formatted with three
    columns: :code:`link_id`, :code:`item_id`, and
    :code:`score`. If score is not provided, it we be
    defaulted to one. :code:`link_id` represents to the
    "user" and `item_id` represents the "item" in the context
    of traditional collaborative filtering userXitem speak.
    """

    def __init__(self, infiles, **kwargs):
        self._inputs = infiles

    def _normalize(self, df):
        if 'score' not in df.columns:
            df['score'] = np.ones(df.shape[0])

        return df[['link_id', 'item_id', 'score']]

    def transform(self, **kwargs):
        """
        Read in files from the :code:`infiles` array given
        upon instantiation.
        """
        dfs = [pd.read_csv(inp) for inp in self._inputs]
        dfs = [self._normalize(df) for df in dfs]

        return pd.concat(dfs), kwargs


class WriteTransform(Transform):
    """
    Write output to disk in csv or json formats.

    Parameters:
        `outfile`: A string that is a path to the desired
        output on the file system.

        `file_format`: A string that is a file extension,
        either :code:`json` or :code:`csv`.
    """

    def __init__(self, outfile, file_format='csv',
                 enc=None, link_key='link_id'):
        self.outfile = outfile
        self.file_format = file_format
        self.link_key = link_key
        self.encoding = enc

    def transform(self, embeddings, **kwargs):
        if self.file_format == 'csv':
            embeddings.to_csv(self.outfile, encoding=self.encoding)
        else:
            with open(self.outfile, 'w+') as f:
                import json
                for row in embeddings.iterrows():
                    f.write(json.dumps({
                        self.link_key: row[0],
                        'embedding': row[1].tolist()
                    }))
                    f.write('\n')

        return embeddings, kwargs
