import numpy as np
import pandas as pd
import types
import warnings

from hidi.transform import Transform
from hidi.linalg import dot
from pyvalid import accepts
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# Catch annoying warnings from nimfa
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nimfa as nf


class ApplyTransform(Transform):
    """
    Apply a function to an input.

    Takes a single argument, `fn`, which must be a function
    accepting one argument (the function to apply), and kwargs.
    """
    def __init__(self, fn):
        self.fn = fn

    def transform(self, x, **kwargs):
        return self.fn(x, **kwargs), kwargs


class SimilarityTransform(Transform):
    """
    Takes the dot product of a link*item matrix.

    Returns either a link*link or item*item similarity matrix.
    If axis is :code:`0`, an item*item matrix is returned, if
    axis is :code:`1` a link*link matrix is returned.
    The returned matrix represents a similarity matrix.

    The transform function returns a tuple containing the
    similarity matrix, and the links or items, depending on
    the axis.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def transform(self, M, items, links, **kwargs):
        M_T = M.transpose()

        if self.axis == 0:
            sim_matrix = dot(M_T, M)
            sim_axis = items
        elif self.axis == 1:
            sim_matrix = dot(M, M_T)
            sim_axis = links
        else:
            raise Exception('Axis must be either 0 or 1')

        return sim_matrix, self.merge_kwargs(dict(items=sim_axis), kwargs)


class ScalarTransform(Transform):
    """
    Scale the matrix using a function or class method.

    `ScalerTransform` takes an `fn` argument that specifies the
    function that should be applied to the matrix. If `fn` is a string
    the scaler transform will try to call a function by that name on
    the matrix, if it is a function reference, scaler transform will
    call that function with the matrix as input.
    """

    def __init__(self, fn=np.log):
        self.fn = fn

    def scale(self, matrix):
        if isinstance(self.fn, types.FunctionType):
            return self.fn(matrix)
        elif type(self.fn) == str:
            return getattr(matrix, self.fn)()

        raise Exception('%s is not a valid scaling function' % self.fn)

    def transform(self, matrix_to_scale, **kwargs):
        """
        Takes a :code:`matrix_to_scale` as a numpy ndarray-like object
        and performs scaling on it, then returns the result.
        """
        out = self.scale(matrix_to_scale)

        return out, kwargs


class SparseTransform(Transform):
    """
    Make a sparse item*link matrix using SciPy's sparse compressed
    row matrix implementation.
    """

    @accepts(object, pd.DataFrame)
    def transform(self, df, **kwargs):
        """
        Takes a dataframe that has :code:`link_id`, :code:`item_id` and
        :code:`score` columns.

        Returns a SciPy :code:`csr_matrix`.
        """
        link_u = list(df.link_id.unique())
        item_u = list(df.item_id.unique())
        data = df.score.as_matrix()

        row = df.link_id.astype('category', categories=link_u).cat.codes
        col = df.item_id.astype('category', categories=item_u).cat.codes

        outshape = (len(link_u), len(item_u))
        in_tuple = (data, (row, col))
        kwargs = self.merge_kwargs(dict(links=link_u, items=item_u), kwargs)

        return csr_matrix(in_tuple, shape=outshape), kwargs


class DenseTransform(Transform):
    """
    Transform a sparse matrix to its dense representation.
    """
    def transform(self, M, **kwargs):
        return M.todense(), kwargs


class ItemsMatrixToDFTransform(Transform):
    """
    Create a Pandas DataFrame object with items as the index.
    """
    def transform(self, M, items, **kwargs):
        """
        Takes a numpy ndarray-like object and a list of item identifiers
        to be used as the index for the DataFrame.
        """
        return pd.DataFrame(M, index=items), kwargs


class KerasEvaluationTransform(Transform):
    """
    Generalized transform for Keras algorithm
    """
    def __init__(self, keras_model, validation_matrix, tts_seed=42,
                 tt_split=0.25, **keras_kwargs):
        self.keras_model = keras_model
        # seed, epochs, batch_size, verbose, cross_validation(boolean)
        self.keras_kwargs = keras_kwargs
        # labeled dataset for modeling and evaluation: item, labels
        self.validation_matrix = validation_matrix
        self.tts_seed = tts_seed
        self.tt_split = tt_split

        if 'item_id' in validation_matrix.columns:
            self.validation_matrix.set_index('item_id', inplace=True)

    def transform(self, M,  **kwargs):
        """
        Takes a numpy ndarray-like object and applies a Keras model to it.
        """
        # clean data
        rows, columns = M.shape
        embedding = M.merge(self.validation_matrix, left_index=True,
                            right_index=True)
        embedding = embedding.values

        # split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            embedding[:, :columns], embedding[:, columns:],
            random_state=self.tts_seed, test_size=self.tt_split)

        self.keras_model.fit(
            x_train, y_train, validation_data=[x_test, y_test],
            **self.keras_kwargs)

        return self.keras_model, kwargs


class KerasPredictionTransform(Transform):
    """
    Generalized transform for Keras algorithm
    """
    def __init__(self, model):
        self.model = model
        # the model is the model output from KerasEvaluationTransform

    def transform(self, M,  **kwargs):
        """
        Takes a numpy ndarray-like object and applies a SkLearn
        algorithm to it.
        """
        predictions = self.model.predict(M)  # M is the ndarray-like object
        return predictions, kwargs


class SkLearnTransform(Transform):
    """
    Generalized transform for SciKit Learn algorithms.

    This transform takes a SciKit Learn algorithm, and its
    keyword arguments upon initialization. It applies the
    algorithm to the input when :code:`transform` is called.

    The algorithm to be applied is likely, but not necessarily
    a :code:`sklearn.decomposition` algorithm.
    """
    def __init__(self, SkLearnAlg, **sklearn_args):
        self.SkLearnAlg = SkLearnAlg
        self.sklearn_args = sklearn_args

    def transform(self, M, **kwargs):
        """
        Takes a numpy ndarray-like object and applies a SkLearn
        algorithm to it.
        """
        sklearn_alg = self.SkLearnAlg(**self.sklearn_args)
        transformed = sklearn_alg.fit_transform(M)
        kwargs['sklearn_fit'] = sklearn_alg

        return transformed, kwargs


class SVDTransform(SkLearnTransform):
    """
    Perform Truncated SVD on the matrix.

    This uses SciKit Learn's Tuncated SVD implementation, which
    is documented here:
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

    All kwargs given to :code:`SVDTransform`'s initialization
    function will be given to :code:`sklearn.decomposition.TruncatedSVD`.

    Please reference the `sklearn docs
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_
    when using this transform.
    """
    def __init__(self, **svd_kwargs):
        super(SVDTransform, self).__init__(TruncatedSVD, **svd_kwargs)


class NimfaTransform(Transform):
    """
    Generalized Nimfa transform.

    This transform takes a nimfa algorithm, and its keyword
    arguments upon initialization. It applies the algorithm
    to the input when :code:`transform` is called.
    """
    def __init__(self, NimfaAlg, **nimfa_kwargs):
        self.NimfaAlg = NimfaAlg
        self.nimfa_kwargs = nimfa_kwargs

    def transform(self, M, **kwargs):
        nimfa_alg = self.NimfaAlg(M, **self.nimfa_kwargs)
        nimfa_fit = nimfa_alg()
        kwargs['nimfa_fit'] = nimfa_fit

        return nimfa_fit.basis(), kwargs


class SNMFTransform(NimfaTransform):
    """
    Perform Sparse Nonnegative Matrix Factorization.

    This wraps nimfa's snmf function, which is documented here:
    http://nimfa.biolab.si/nimfa.methods.factorization.snmf.html

    All kwargs given to :code:`SNFMTransform`'s initialization
    function will be given to :code:`nimfa.Snmf`.

    Please reference the `nimfa docs
    <http://nimfa.biolab.si/nimfa.methods.factorization.snmf.html>`_
    when using this transform.
    """

    def __init__(self, **snmf_kwargs):
        super(SNMFTransform, self).__init__(nf.Snmf, **snmf_kwargs)
