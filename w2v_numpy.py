import random
from typing import Sequence, Tuple, List
import numpy.typing as npt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(logits: npt.NDArray) -> npt.NDArray:
    """Input dim: [V, B]"""
    x = logits - logits.max(axis=0)  # max each batch separately
    exp = np.exp(x)
    return exp / exp.sum(axis=0)


def grouper(iterable, batch_size):
    """Collect data into fixed-length chunks or blocks."""
    args = [iter(iterable)] * batch_size
    # Note, zip will discard incomplete batches at the end of iterable.
    return zip(*args)


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

        Input dims:
            cbow.forward: [N-1, B]
            sgram.forward: [1, B]
        """
        # If a word is not in the vocab, it gets ignored.
        idx = self.encode_idx_fast(tokens)
        ohe_mat = np.zeros((self.size, tokens.shape[1]))
        for i, batch_idx in enumerate(idx.T):
            index, counts = np.unique(batch_idx, return_counts=True)
            ohe_mat[index, i] = counts
        return ohe_mat  # dims: [vocab.size, batch_size]

    def encode_ohe_fast_single_word(self, tokens: npt.NDArray[str]):
        """Encodes from tokens to flat OHE vector.

        More efficient for the case that there is only a single word per OHE.
        E.g. for target word encoding, or context in the case of S-gram.

        For multiword per OHE, possible duplicate words in OHE necessitate other
        OHE method.

        Input dims: [1, Batch]
        """
        assert tokens.shape[0] == 1, ('Method only valid on single word OHE.'
                                      ' Multiword OHE method required')
        # If a word is not in the vocab, it gets ignored.
        idx = self.encode_idx_fast(tokens)
        ohe_mat = np.zeros((self.size, tokens.shape[1]))  # [V, B]
        ohe_mat[idx.squeeze(), range(tokens.shape[1])] = 1
        return ohe_mat  # dims: [V, B]

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


class Cbow:
    """Continuous Bag of Words model from original word2vec paper.

    To run the Skip gram model, use the Sgram class and pass the target as input
    and context into the loss function.

    E.g.  CBOW                  Skip Gram
    cbow = Cbow(...)            sgram = Sgram(...)
    cbow.forward(context)       sgram.forward(target)
    cbow.loss(target)           sgram.loss(context)
    cbow.backward()             sgram.backward()
    cbow.optim_sgd(alpha)       sgram.optim_sgd(alpha)

    Efficient Estimation of Word Representations in Vector Space.
    https://arxiv.org/pdf/1301.3781
    """

    @property
    def window(self):
        # context length. E.g. for window of 5, we use 2 words before and 2
        # words after.
        return self._window

    @property
    def embed_dim(self):
        # number of embedding dimensions
        return self._embed_dim

    @property
    def batch_size(self):
        # size of batch. This initializes arrays. In some cases, we iterate
        # through each batch item individually.
        return self._batch_size

    def __init__(
            self, vocab: Vocab, embed_dim: int, window: int,
            batch_size: int, seed=None):

        np.random.seed(seed)
        v = 1 / np.sqrt(vocab.size)
        d = 1 / np.sqrt(embed_dim)
        self.params = {
            'w1': np.random.uniform(-v, +v, size=(embed_dim, vocab.size)),
            'w2': np.random.uniform(-d, +d, size=(vocab.size, embed_dim))
        }

        # stats data frame tracks the progress of training.
        self.stats = pd.DataFrame(
            columns=[
                'epoch', 'batch_iters', 'iters', 'time (min)', 'loss', 'alpha'
            ]).astype({
                'epoch': 'int', 'batch_iters': 'int', 'iters': 'int',
                'time (min)': 'float64', 'loss': 'float64', 'alpha': 'float32'
            })
        self.loss = []  # history of loss
        self.epoch = 0

        self._window = window
        self._embed_dim = embed_dim
        self._batch_size = batch_size
        self.vocab = vocab
        # State variables, used in backprop.
        self.state = {
            'context': np.empty((window-1, batch_size), dtype=np.dtype('<U20')),
            'context_ohe': np.zeros((vocab.size, batch_size)),
            'projection': np.zeros((embed_dim, batch_size)),
            'logits': np.zeros((vocab.size, batch_size)),
            'probs': np.zeros((vocab.size, batch_size)),
        }
        self.grads = {
            'w1': np.zeros_like(self.params['w1']),
            'w2': np.zeros_like(self.params['w2'])
        }

    def forward(self, context: npt.NDArray[str]) -> npt.NDArray:
        """Input dim: [N-1, B]."""
        context_ohe = self.vocab.encode_ohe_fast(context) / (self.window-1)  # [V, B]
        projection = self.params['w1'] @ context_ohe  # [D,V] @ [V,1] = [D,B]
        logits = self.params['w2'] @ projection  # [V, D] x [D, 1] = [V, B]
        probs = softmax(logits)  # [V, B]
        # Save the forward pass state.
        self.state['context'] = context
        self.state['context_ohe'] = context_ohe  # used by self.backward
        self.state['projection'] = projection
        self.state['probs'] = probs
        return probs  # [V, B]

    def backward(self):
        # dlogits calculation is abstracted out as it differs slightly for CBOW
        # and S-gram.
        dlogits = self._calc_dlogits()
        self.grads['w2'] = dlogits @ self.state['projection'].T  # [V,B] @ [B,D] = [V,D]
        dproj = self.params['w2'].T @ dlogits  # [D,V] @ [V,B] = [D,B]
        self.grads['w1'] = dproj @ self.state['context_ohe'].T  # [D,V] = [D,B] @ [B,V]

    def loss_fn(self, actual: npt.NDArray):
        """Loss function differs for CBOW and S-gram"""
        # Cbow input dims: [1, B]
        self.state['target_str'] = actual
        self.state['target_ohe'] = self.vocab.encode_ohe_fast_single_word(actual)
        self.state['loss'] = self.cross_entropy()
        return self.state['loss']

    def optim_sgd(self, alpha):
        self.params['w2'] -= alpha * self.grads['w2']
        self.params['w1'] -= alpha * self.grads['w1']

    def _calc_dlogits(self):
        """dlogits calculation differs for S-gram and Cbow."""
        # CBOW implementation of dlogits.
        return self.state['probs'] - self.state['target_ohe']  # [V,B] - [V,B] = [V,B]

    def cross_entropy(self):
        """Cross entropy loss differs for S-gram and CBOW."""
        # sum across vocab [V], mean across batch [B],
        return -(np.log(self.state['probs'] + 1e-9) * self.state['target_ohe']).sum(0).mean()

    def forward_quick(self, context: npt.NDArray[str]):
        """Equivalent to `forward`, but supposed to be quicker.

        Subsets matrix columns corresponding to input context so that a smaller
        matrix is used.

        Turns out to be marginally slower than forward when using batch=512, so
        this method is not used.

        Input dim: [N-1, B]."""
        context_idx = self.vocab.encode_idx_fast(context)  # [N, B]
        projection_sub = self.params['w1'][:, context_idx]  # [D, V] -> [D, N, B]
        projection = projection_sub.mean(axis=1, keepdims=True).squeeze(axis=1)  # [D, N, B] -> [D, B]
        logits = self.params['w2'] @ projection  # [V, D] x [D, B] = [V, B]
        probs = softmax(logits)  # [V, B]
        # Save the forward pass state.
        self.state['context'] = context
        self.state['context_ohe'] = self.vocab.encode_ohe_fast(context)  # / (self.window-1)   # used by self.backward
        self.state['probs'] = probs
        self.state['projection'] = projection
        return probs

    def backward_quick(self):
        """Equivalent to backward().

        Speeds up by about 25% for batch_size=512 by operating on relevant
        columns only.
        """
        dlogits = self._calc_dlogits()
        self.grads['w2'] = dlogits @ self.state['projection'].T  # [V,B] @ [B,D] = [V,D]
        dproj = self.params['w2'].T @ dlogits  # [D,V] @ [V,B] = [D,B]

        # slice out each batch and insert only the changed w1 values.
        context_idx = self.vocab.encode_idx_fast(self.state['context'])  # [N-1, B]
        # Can save 30 microsecs by subsetting on context_ohe within the loop.
        context_reduced = np.take_along_axis(
            self.state['context_ohe'], context_idx, axis=0)  # [N-1, B]
        self.grads['w1'] = np.zeros_like(self.grads['w1'])
        for b in range(self.batch_size):
            self.grads['w1'][:, context_idx[:, b]] += dproj[:, [b]] @ context_reduced[:, [b]].T

    def plot_loss_curve(self, filename=None, ma=100):
        """Plots loss curve to file. Convenient tracking of training progress."""
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.2)  # Remark 1
        ax = fig.add_subplot(111)
        y = np.convolve(
            np.array(self.loss),
            np.ones(min(len(self.loss), ma)) / min(len(self.loss), ma),
            'valid')
        ax.plot(np.arange(len(y)), y, 'blue')
        # log(1/V) is the neutral untrained error. When all probabilities are equal.
        ax.hlines(y=-np.log(1 / self.vocab.size), colors='red', xmin=0, xmax=len(y))
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

    def plot_prediction(self, data, filename=None, target=None,
                        context=None, top_n=5):
        """Plots the predicted probabilities of the context and target.

        target: plot probability for the target word, and its context words. If
            None, a random word will be selected

        top_n: plot top_n probability words as well These will tend to be
        related to the target word on a trained model.
        """
        if target is None or context is None:
            context, target = data.sample_random()
            context = np.expand_dims(np.array(context), axis=1)
        probs = self.forward(context).squeeze()
        top_n_idx = np.argsort(probs)[-top_n:]
        top_n_probs = probs[top_n_idx]
        top_n_labels = self.vocab.vocab[top_n_idx].squeeze()
        context_idx = self.vocab.encode_idx_fast(np.array(context)).squeeze()
        context_probs = probs[context_idx]
        target_prob = probs[np.squeeze(self.vocab.vocab == target)]
        plot_probs = np.concatenate([target_prob, context_probs, top_n_probs])
        plot_labels = np.append(np.append(target, context), top_n_labels)
        plot_colours = ['red' if label == target else 'blue' for label in plot_labels]

        fig = plt.figure()
        fig.subplots_adjust(bottom=0.2)  # Remark 1
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(plot_probs)), plot_probs, color=plot_colours)
        ax.set_title(f'target: {target}, context: {context.squeeze().tolist()}')
        ax.ticklabel_format(style='plain')
        ax.set_xticks(np.arange(len(plot_probs)))
        ax.set_xticklabels(plot_labels, rotation=80)
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close('all')
        else:
            plt.show()


class Sgram(Cbow):

    def _calc_dlogits(self):
        """dlogits calculation differs for S-gram and Cbow."""
        # For Sgram we have to mean across all the output context words.
        return self.state['probs'] - self.state['target_ohe'].mean(2)  # [V,B] - [V,B] = [V,B]

    def cross_entropy(self):
        """Cross entropy loss differs for S-gram and CBOW."""
        # Sgram has an extra dim in the output as each context word in output
        # has its own OHE.
        # dims: probs: [V, B], target_ohe: [V, B, N - 1]
        # sum across vocab, mean across context words and batch.
        return -(np.log(self.state['probs'] + 1e-9)[:, :, np.newaxis] *
                 self.state['target_ohe']).sum(0).mean()

    def loss_fn(self, actual: npt.NDArray):
        """Input dim for Sgram: [N-1, B]"""
        self.state['target_str'] = actual
        ohe_target = []
        for i, c in enumerate(actual):  # enumerate across context words (N-1)
            ohe_target.append(self.vocab.encode_ohe_fast_single_word(np.expand_dims(c, axis=0)))
        self.state['target_ohe'] = np.stack(ohe_target, axis=2)
        self.state['loss'] = self.cross_entropy()
        return self.state['loss']
