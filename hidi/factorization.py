import warnings
import numpy as np
import collections
from numpy.random import permutation
from sklearn.decomposition import TruncatedSVD
from hidi.transform import Transform

# Catch annoying warnings from nimfa
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nimfa as nf


class W2VStringTransform(Transform):
    """
    Takes a pandas Dataframe  and transforms it into a
    string

    :param n_shuffles: The number of suffles for the
    `item_id`.
    :type n_shuffles: int
    """
    def __init__(self, n_shuffles=3, **w2v_kwargs):
        self.w2v_kwargs = w2v_kwargs
        self.n_shuffles = n_shuffles

    def transform(self, df, **kwargs):
        """
        :param df: a pandas Dataframe with two columns:
        :code:`link_id`, :code:`item_id`
        :type df: pandas.Dataframe
        :rtype: str
        """
        if 'item_id' not in df.index:
            df.set_index('item_id', inplace=True)

        words = ''
        for index in df.index.unique():
            a0 = df.loc[index].link_id
            a = a0
            for i in range(self.n_shuffles-1):
                a = np.append(a, permutation(a0))
            b = " ".join(str(x) for x in a)
            words = " ".join([words, b]).strip()
        return words, kwargs


class W2VBuildDatasetTransform(Transform):
    def __init__(self, vocabulary_size=5000, **w2v_kwargs):
        self.vocabulary_size = vocabulary_size
        self.w2v_kwargs = w2v_kwargs

    def transform(self, words, **kwargs):
        if isinstance(words, str):
            words = words.split()
        count = [['UNK', -1]]
        count_words = collections.Counter(words)
        count.extend(count_words.most_common((self.vocabulary_size-1)))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary
        return (data, count, dictionary, reverse_dictionary), kwargs


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
