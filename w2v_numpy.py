from typing import Sequence, Tuple, List
import numpy.typing as npt

import numpy as np



class Vocab:
    """Generates vocabulary and encoding/decoding utils from raw train_data.

    For encoding/decoding methods, assumes input dims are 3 long. E.g.
    [Vocab.size, embed_dim, batch_size]. Include batch dim but set to 1 if not
    batching.
    """

    @property
    def vocab(self):
        return self._vocab

    @property
    def size(self):
        return self._size

    @property
    def index(self):
        return self._index


    @property
    def token2index(self):
        return self._token2index

    def __init__(self, data: Sequence[Sequence[str]]):
        vocab = set(' '.join([' '.join(line) for line in data]).split(' '))

        self._vocab = np.expand_dims(np.array(list(vocab)), 1)  # dims: [V, 1]
        self._vocab.sort()
        self._size = len(self._vocab)
        self._index = np.expand_dims(np.arange(self._size), 1)  # dims: [V, 1]
        self._token2index = dict(zip(self._vocab.squeeze(),
                                     self._index.squeeze()))

    def encode_ohe(self, tokens: npt.NDArray[str]):
        """Encodes from tokens to flat OHE vector.

        Input dims: [Context length - 1, Batch]
        """
        # If a word is not in the vocab, it gets ignored.
        token_mat = np.zeros((self.size, tokens.shape[1]))
        for i, batch in enumerate(tokens.T):
            # we have to generate ohe for each word and add them. If a word occurs twice within the context they will
            # get a value of 2 in there position of the OHE vector. This means it will get double the weight in the
            # projection layer, which has the effect of averaging all words, including duplicate instances in the
            # projection layer.
            for word in batch:
                token_mat[:, i] += np.isin(self.vocab, word).astype('float').squeeze()
        return token_mat  # dims: [vocab.size, batch_size]

    def encode_ohe_fast(self, tokens: npt.NDArray[str]):
        """Encodes from tokens to flat OHE vector.

        Input dims: [Context length - 1, Batch]
        """
        # If a word is not in the vocab, it gets ignored.
        idx = self.encode_idx_fast(tokens)
        ohe_mat = np.zeros((self.size, tokens.shape[1]))

        for i, batch_idx in enumerate(idx.T):
            index, counts = np.unique(batch_idx, return_counts=True)
            ohe_mat[index, i] = counts
        return ohe_mat  # dims: [vocab.size, batch_size]


    def encode_ohe_ignore_dupe(self, tokens: npt.NDArray[str]):
        """Encodes from tokens to flat OHE vector.

        Input dims: [Context length - 1, Batch]
        """
        # If a word is not in the vocab, it gets ignored. Repeated words in the
        # same context window are also ignored.
        token_mat = [np.isin(self.vocab, batch, assume_unique=True).astype(
            'float') for batch in tokens.T]
        token_mat = np.concatenate(token_mat, axis=1)
        return token_mat  # dims: [vocab.size, batch_size]

    def encode_idx(self, tokens: npt.NDArray[str]):
        """From text tokens to index numbers.

        Input dims: [Context length - 1, Batch]
        """
        # If a word is not in the vocab, it gets ignored.
        # Need to iterate through tokens to handle repeated words within
        # a batch.
        index_mat = np.zeros(tokens.shape, dtype=self._index.dtype)
        for i, batch in enumerate(tokens.T):
            index_mat[:, i] = np.concatenate([self._index[self.vocab == word] for word in batch])
        return index_mat  # dims -> [window-1, batch_size]

    def encode_idx_fast(self, tokens: npt.NDArray[str]):
        """From text tokens to index numbers.

        Input dims: [Context length - 1, Batch]
        """
        # This should keep repeat tokens in a context window.
        idx = [[self._token2index[token] for token in batch] for batch in tokens.T]
        return np.vstack(idx).T  # output shape: [Context length - 1, batch]



