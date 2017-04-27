import numpy as np
import pandas as pd
import types
import warnings

from hidi.transform import Transform
from hidi.linalg import dot
from pyvalid import accepts
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# Catch annoying warnings from nimfa
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nimfa as nf


class ApplyTransform(Transform):
    """
    Apply a function to an input.

    Takes a single argument, `fn`, which must be a function
    accepting one argument (the function to apply), and kwargs.

    :param fn: The function to be applied to transform input.
    :type fn: function
    """
    def __init__(self, fn):
        self.fn = fn

    def transform(self, x, **kwargs):
        """
        :param x: The input to the function :code:`fn`.
        """
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

    :param axis: The axis to perform the dot product for.
    :type axis: int[0,1]
    """

    def __init__(self, axis=0):
        self.axis = axis

    def transform(self, M, items, links, **kwargs):
        """
        :param M: The matrix to create a similarity matrix from
        :type M: numpy ndarray-like

        :param items: Array of :code:`item_ids` in the same order
            that they appear in :code:`M`.
        :type items: array

        :param links: Array of :code:`link_ids` in the same order
            that they appear in :code:`M`.
        :type links: array
        """
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

    :param fn: The scalar function to use. If :code:`fn` is a string
        then an attribute of that name will be looked up and called.
        If :code:`fn` is a function, that function will be called
        with the input given to transform.
    :type fn: str | function
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

        :param df: The DataFrame to make a sparse matrix from. Must have
            :code:`link_id`, :code:`item_id`, and :code:`score` columns.
        :type df: pandas.DataFrame
        :rtype: scipy.sparse.csr_matrix
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
        """
        Takes a sparse matrix and transform it into its dense representation

        :param M: a sparse matrix
        :type M: scipy.sparse classes
        """
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

    This transform takes a Keras sequential model, a validation matrix and
    its keyword arugments upon initialization.

    :param keras_model: a Keras sequential model which is documented here:
        https://keras.io/getting-started/sequential-model-guide/
    :type keras_model: Keras Sequential model
    :param validation_matrix: A validation matrix is a dataframe that has
        :code:`item_id` index, other 'label' columns. It will be inner
        joined with the M matrix and then fed into the Keras sequential
        model.
    :type validation_matrix: pandas.DataFrame
    :param tts_seed: random state seed for :code:`train_test_split`
    :type tts_seed: int
    :param tt_split: the proportion of the dataset to include in the test
        split for :code:`train_test_split`
    :type tt_split: float
    """

    def __init__(self, keras_model, validation_matrix, tts_seed=42,
                 tt_split=0.25, **keras_kwargs):
        self.keras_model = keras_model
        self.keras_kwargs = keras_kwargs
        self.validation_matrix = validation_matrix
        self.tts_seed = tts_seed
        self.tt_split = tt_split

        if 'item_id' in validation_matrix.columns:
            self.validation_matrix.set_index('item_id', inplace=True)

    def transform(self, M,  **kwargs):
        """
        Takes a Takes a dataframe that has :code:`item_id` index, other
        'features' columns for prediction, and applies a Keras sequential
        model to it.

        :param M: a dataframe that has :code:`item_id` index, other
        'features' columns
        :type M: pandas.DataFrame
        :rtype: a tuple with trained Keras model and its keyword
            arguments

        """
        rows, columns = M.shape
        embedding = M.merge(self.validation_matrix, left_index=True,
                            right_index=True)
        embedding = embedding.values

        x_train, x_test, y_train, y_test = train_test_split(
            embedding[:, :columns], embedding[:, columns:],
            random_state=self.tts_seed, test_size=self.tt_split)

        self.keras_model.fit(
            x_train, y_train, validation_data=[x_test, y_test],
            **self.keras_kwargs)

        return self.keras_model, kwargs


class KerasKfoldTransform(Transform):
    """
    Generalized transform for Keras algorithm with k fold cross validation
    evaluation

    :param keras_model: a Keras sequential model which is documented here:
        https://keras.io/getting-started/sequential-model-guide/
    :type keras_model: Keras Sequential model
    :param validation_matrix: A validation matrix is a dataframe that has
        :code:`item_id` index, other 'label' columns. It will be inner
        joined with the M matrix and then fed into the Keras sequential
        model.
    :type validation_matrix: pandas.DataFrame
    :param kfold_n_splits: Number of folds for kfold. Must be at least 2.
    :type kfold_n_splits: int
    :param kfold_seed: random state seed for kfold
    :type kfold_seed: None, int or RandomState
    :param kfold_shuffle: Whether to shuffle the data before splitting
        into batches for kfold
    :type kfold_shuffle: boolean
    """
    def __init__(self, keras_model, validation_matrix,
                 kfold_n_splits=10, kfold_seed=42, kfold_shuffle=True,
                 classification=False, **keras_kwargs):
        self.keras_model = keras_model
        self.keras_kwargs = keras_kwargs
        self.validation_matrix = validation_matrix

        self.kfold_n_splits = kfold_n_splits
        self.kfold_seed = kfold_seed
        self.kfold_shuffle = kfold_shuffle

        self.classification = classification

        if 'item_id' in validation_matrix.columns:
            self.validation_matrix.set_index('item_id', inplace=True)

    def transform(self, M,  **kwargs):
        """
        Takes a Takes a dataframe that has :code:`item_id` index, other
        'features' columns for prediction, and applies a Keras sequential
        model to it.

        :param M: a dataframe that has :code:`item_id` index, other
            'features' columns
        :type M: pandas.DataFrame
        :rtype: a tuple with trained Keras model and its keyword
            arguments
        """
        rows, columns = M.shape
        embedding = M.merge(self.validation_matrix, left_index=True,
                            right_index=True)
        embedding = embedding.values

        if self.classification:
            kfold = StratifiedKFold(n_splits=self.kfold_n_splits,
                                    random_state=self.kfold_seed,
                                    shuffle=self.kfold_shuffle)
        else:
            kfold = KFold(n_splits=self.kfold_n_splits,
                          random_state=self.kfold_seed,
                          shuffle=self.kfold_shuffle)

        X = embedding[:, :columns]
        Y = embedding[:, columns:]
        for train_index, test_index in kfold.split(X, Y):
            self.keras_model.fit(
                X[train_index], Y[train_index],
                validation_data=[X[test_index], Y[train_index]],
                **self.keras_kwargs)

        return self.keras_model, kwargs


class KerasPredictionTransform(Transform):
    """
    Generalized transform for Keras model prediction

    This transform takes a trained Keras model. It applies the train model
    to the input when :code:`transform` is called.

    :param: model: trained keras model
    :type M: trained keras model
    """
    def __init__(self, model):
        self.model = model

    def transform(self, M,  **kwargs):
        """
        Takes a numpy ndarray-like object and applies a trained Keras model
        to it.

        Returns the predictions from the trained Keras model

        :param: M: a dataframe that has :code:`item_id` index, other
            'features' columns
        :param: M: pandas.DataFrame
        :rtype: ndarray-like object with its kwargs
        """
        predictions = self.model.predict(M)
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
