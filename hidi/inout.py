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
    of traditional collaborative filtering.

    :param infiles:
        Array of paths to csv documents to be loaded
        and concatenated into one DataFrame. Each csv
        document must have a :code:`link_id` and a
        :code:`item_id` column. An optional
        :code:`score` column may also be supplied.
    :type infiles: array
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

    :param outfile: A string that is a path to the desired
        output on the file system.
    :type outfile: str

    :param file_format: A string that is a file extension,
        either :code:`json` or :code:`csv`.
    :type file_format: str
    """

    def __init__(self, outfile, file_format='csv',
                 enc=None, link_key='link_id'):
        self.outfile = outfile
        self.file_format = file_format
        self.link_key = link_key
        self.encoding = enc

    def transform(self, df, **kwargs):
        """
        Write a DataFrame to a file.

        :param df: The Pandas DataFrame to be written to a
            file
        :type df: pandas.DataFrame
        """
        if self.file_format == 'csv':
            df.to_csv(self.outfile, encoding=self.encoding)
        else:
            with open(self.outfile, 'w+') as f:
                import json
                for row in df.iterrows():
                    f.write(json.dumps({
                        self.link_key: row[0],
                        'embedding': row[1].tolist()
                    }))
                    f.write('\n')

        return df, kwargs
