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


class Dataloader:
    """Iterator for the train_data object.

    Generates target token and context tokens around the target word. Each
    iteration increments along the training_data to generate a context, target
    pair.
    """

    def __init__(self, data: Sequence[Sequence[str]], window: int):
        assert (window - 1) * 0.5 % 1 == 0 and window > 2, f'window must be odd and > 2'
        self.data = data
        self.window = int(window)
        self.line_no = int(0)  # line number the iterator is at.
        self.win_pad = int((window - 1) / 2)  # padding on either side of target
        self.idx = self.win_pad  #
        self.line = self.data[self.line_no]

    def __iter__(self):
        return self

    def __next__(self):
        context, target = self._data_sampler()
        self._index_incrementer()
        return context, target

    def _data_sampler(self) -> Tuple[List[str], str]:
        """Generates context and target given an index (idx) to target."""
        start_idx = int(self.idx - self.win_pad)
        end_idx = int(self.idx + self.win_pad)
        window_idx = list(range(start_idx, end_idx + 1))
        window_idx.remove(self.idx)
        context = np.array(self.line)[window_idx].tolist()
        target = self.line[self.idx]
        return context, target

    def _index_incrementer(self):
        """Increments to the next index (idx)."""
        # This will not include partial windows at the beginning and end of
        # lines.
        if self.idx + self.win_pad + 1 < len(self.line) - 1:
            self.idx += 1
        elif self.line_no < len(self.data) - 1:
            self.idx = self.win_pad
            self.line_no += 1
            self.line = self.data[self.line_no]
            # if next line is < window length, need to go to next line.
            while len(self.line) < self.window:
                self.line_no += 1
                self.line = self.data[self.line_no]
        else:
            raise StopIteration

    def sample_random(self, seed=None):
        """Returns random target and context words."""
        random.seed(seed)
        line = []
        while len(line) < self.window:
            line = self.data[random.randint(0, len(self.data))]
        idx = random.randint(self.win_pad, len(line) - self.win_pad - 1)
        target = line[idx]
        context = line[(idx - self.win_pad):(idx + self.win_pad + 1)]
        context.remove(target)
        return context, target

    def sample_current(self):
        """Returns the current values without iterating forward."""
        return self._data_sampler()

    def __len__(self):
        if hasattr(self, 'len'):
            return self.len
        else:
            length = 0
            for line in self.data:
                length += max(0, (len(line) - self.window + 1))
            self.len = length
            return length

import random
