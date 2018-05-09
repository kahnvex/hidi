import warnings
import numpy as np
import pandas as pd
import collections
from numpy.random import permutation
import random
from sklearn.decomposition import TruncatedSVD
from hidi.transform import Transform

# Catch annoying warnings from nimfa
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nimfa as nf


class W2VStringTransform(Transform):
    """
    Takes a pandas Dataframe and transforms it into a
     list of sentences. Each sentence is a list
     of words/items (unicode strings) that will be used for
     training.

    :param n_shuffles: The number of suffles for the
    `item_id`.
    :type n_shuffles: int
    """
    def __init__(self, n_shuffles=3, **w2v_kwargs):
        self.w2v_kwargs = w2v_kwargs
        self.n_shuffles = n_shuffles

    def transform(self, df_temp, **kwargs):
        """
        :param df: a pandas Dataframe with two columns:
        :code:`link_id`, :code:`item_id`
        :type df: pandas.Dataframe
        :rtype: a list sentences and each sentence is a list
            of shuffled item_id
        """
        df = df_temp
        if ('link_id' in df.index) | ('item_id' in df.index):
            df.reset_index(inplace=True)

        df.set_index('link_id', inplace=True)
        sentences = []
        for index in df.index.unique():
            joined_shuffled_list = []
            temp_item_id_list = df.loc[index].item_id
            item_id_list = temp_item_id_list
            for i in range(self.n_shuffles - 1):
                try:
                    item_id_list = np.append(item_id_list,
                                             permutation(temp_item_id_list))
                except:
                    item_id_list = np.append(item_id_list, temp_item_id_list)
            joined_shuffled_list = list(np.append(joined_shuffled_list,
                                                  item_id_list))
            sentences.append(joined_shuffled_list)
        return sentences, kwargs


class W2VGenismTransform(Transform):
    """
    Generalized transform for gensim.models.Word2Vec
    Takes an uninitialized :code:`gensim.models.Word2Vec`.
    Read more about this model:
    https://radimrehurek.com/gensim/models/word2vec.html
    Note that the uninitialized gensim.model.Word2Vec model can be created
    without sentences.

    :param gensim_w2v_model: an uninitialized gensim.model.Word2Vec model
    :type gensim_w2v_model: gensim.model.Word2Vec
    """
    def __init__(self, gensim_w2v_model, **gensim_w2v_kwargs):
        self.gensim_w2v_kwargs = gensim_w2v_kwargs
        self.gensim_w2v_model = gensim_w2v_model

    def transform(self, sentences, **kwargs):
        """
        Takes a string of items

        :param sentences: The sentences iterable can be simply a list.
            Each sentence is a list of words (unicode strings)
            that will be used for training.
        :type sentences: str or other type of iterables
        :rtype: a trained gensim.model.Word2Vec
        """
        self.gensim_w2v_model.build_vocab(sentences, **self.gensim_w2v_kwargs)
        return self.gensim_w2v_model, kwargs


class W2VGensimToDFTransform(Transform):
    """
    Takes a trained gensim.model.Word2Vec model and save the item embeddings to
    pandas.Dataframe

    :param trained_gensim_w2v_model: a trained :code:`gensim.model.Word2Vec`
    :type trained_gensim_w2v_model: gensim.model.Word2Vec
    """
    def __init__(self, trained_gensim_w2v_model, **gensim_w2v_kwargs):
        self.gensim_w2v_kwargs = gensim_w2v_kwargs
        self.model = trained_gensim_w2v_model

    def transform(self, **kwargs):
        index = []
        embeddings = []
        for key in self.model.wv.vocab.keys():
            index.append(key)
            embeddings.append(self.model[key])
        embeddings = pd.DataFrame(embeddings, index=index)
        return embeddings, kwargs


class W2VBuildDatasetTransform(Transform):
    """
    Takes a string of list of items(words) and tokenizes it.
    :param vocabulary_size: top n most frequent items(words)
    :type vocabulary_size: int
    """
    def __init__(self, vocabulary_size=50000, **w2v_kwargs):
        self.vocabulary_size = vocabulary_size
        self.w2v_kwargs = w2v_kwargs

    def transform(self, words, **kwargs):
        """
        :param words: a list or a string of items
        :type words: list or str
        :rtype: a tuple of `data`, `count`, `dictionary` and
            `reverse_dictionary` `data` is the tokenized words, `count` is a
            list of tuple which consists of `(item, count)`, `dictionary`
            stores tokens of each items as its keys and items as its values
            and `reverse_dictionary` is the reversed of `dictionary`
        """
        if isinstance(words, str):
            words = words.split()
        count = [['UNK', -1]]
        count_words = collections.Counter(words)
        count.extend(count_words.most_common((self.vocabulary_size - 1)))
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
        kwargs['reverse_dictionary'] = reverse_dictionary
        kwargs['item_count'] = count
        kwargs['vocabulary_size'] = self.vocabulary_size
        return data, kwargs


class W2VGenerateBatchTransform(Transform):
    def __init__(self, batch_size=8, num_skips=2, skip_window=1, **w2v_kwargs):
        self.batch_size = batch_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.w2v_kwargs = w2v_kwargs

    def transform(self, data, **kwargs):
        data_index = 0
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        return (batch, labels), kwargs


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
