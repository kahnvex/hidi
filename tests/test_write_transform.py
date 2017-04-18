import sys
import unittest
import pandas as pd

from io import BytesIO, StringIO
from hidi.inout import WriteTransform


class TestWriteTransform(unittest.TestCase):
    def setUp(self):
        outfile = StringIO() if sys.version_info >= (3, 0) else BytesIO()
        self.df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        writer = WriteTransform(outfile=outfile, file_format='csv')
        writer.transform(self.df)
        outfile.seek(0)
        self.file_df = pd.read_csv(outfile)

    def test_file_is_written(self):
        self.assertEqual(self.file_df.iloc[0, 1], 1)
        self.assertEqual(self.file_df.iloc[1, 2], 5)
