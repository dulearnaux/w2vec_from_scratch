import random
import math
from typing import Sequence, Tuple, List
import numpy.typing as npt

import numpy as np
import pandas as pd
from collections import Counter
from functools import partial
from itertools import batched
import matplotlib.pyplot as plt


def softmax(logits: npt.NDArray) -> npt.NDArray:
    """Input dim: [V, B]"""
    x = logits - logits.max(axis=0)  # max each batch separately
    exp = np.exp(x)
    return exp / exp.sum(axis=0)


def grouper(iterable, batch_size):
    """Collect data into fixed-length chunks or blocks."""
    args = [iter(iterable)] * batch_size
    # Note, zip will discard incomplete batches at the end of iterator.
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

        # Go through data in chunks to balance speed and memory use.
        vocab = set()
        for chunk in batched(data, 100_000):  # 100k batches limits memory.
            vocab = vocab.union(set(' '.join([' '.join(line) for line in chunk]).split(' ')))

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
        np.add.at(ohe_mat, (idx, np.arange(tokens.shape[1])), 1)
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

    Attributes:
        window: full length of context window to use. E.g. if window is 5, we
            sample 2 words to either side of the target word, such that the
            total window length is 2+1+2.
        k: number of negative samples to return. These will be
            selected to not intersect with either target or context.
    """

    def __init__(self, data: Sequence[Sequence[str]], window: int,
                 negative_samples=0):
        assert (window - 1) * 0.5 % 1 == 0 and window > 2, f'window must be odd and > 2'
        self.data = data
        self.window = int(window)
        self.line_no = int(0)  # line number the iterator is at.
        self.win_pad = int((window - 1) / 2)  # padding on either side of target
        self.idx = self.win_pad  #
        self.line = self.data[self.line_no]
        # In rare cases the first line is shorter than window.
        self.short_line_check()
        self.k = negative_samples
        if negative_samples > 0:
            # initialize the noise distribution
            self._init_noise_dist()

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
            self.short_line_check()
        else:
            raise StopIteration

    def short_line_check(self):
        """If line is < window length, need to go to next line."""
        while len(self.line) < self.window:
            self.line_no += 1
            if self.line_no < len(self.data) - 1:
                self.line = self.data[self.line_no]
            else:
                raise StopIteration



    def _init_noise_dist(self):
        """Creates a noise distribution to sample from.

        P ~ U(x)^0.75

        Adopt distribution from Mikolov et al. (pf 4).
        https://arxiv.org/pdf/1310.4546
        """
        # Get word frequencies in the data.
        data_flat = ' '.join([' '.join(line) for line in self.data]).split(' ')
        frequencies = Counter(data_flat)
        total = len(data_flat)
        for k in frequencies.keys():
            frequencies[k] /= total
        assert math.isclose(sum([v for v in frequencies.values()]), 1), \
            'Something is wrong with the noise distribution.'

        for k in frequencies.keys():
            frequencies[k] **= 0.75

        norm_const = sum([v for v in frequencies.values()])
        for k in frequencies.keys():
            frequencies[k] /= norm_const
        assert math.isclose(sum([v for v in frequencies.values()]), 1), \
            'Something is wrong with the noise distribution.'
        self.noise_dist_dict = frequencies
        self.noise_dist_words = np.array([k for k in frequencies.keys()])
        self.noise_dist_probs = np.array([v for v in frequencies.values()])

    def neg_samples(self, input: npt.NDArray[str], output: npt.NDArray[str], seed=None):
        """Generates a batch of negative samples.

        Words in input and output cannot occur in neg_samples within a sample
        but can occur within a batch.

        Input is input layer tokens. Context for CBOW, target for Sgram.
        Output is output layer tokens. Target for CBOW, context for Sgram.
        """

        assert self.k > 0, 'Attempt to draw negative samples when k <= 0 is invalid.'
        rng = np.random.default_rng(seed=seed)
        words_to_exclude = np.concatenate([input, output], axis=0)

        sampler = partial(rng.choice, a=self.noise_dist_words, replace=False, p=self.noise_dist_probs, shuffle=False)
        neg_samples = sampler(size=(self.k, output.shape[1]))
        # Duplicate samples are rare, so this remains about 5x faster than
        # sampling independently for each column to enforce uniqueness.
        neg_samples = self._remove_dupe_negs(sampler, neg_samples, words_to_exclude)
        return neg_samples

    def _remove_dupe_negs(self, generator, samples: npt.NDArray[str],
                          excludes: npt.NDArray[str]):
        """Replaces samples with duplicate words."""
        # dims samples:[K, B], excludes:[N, B]
        for b in range(samples.shape[1]):  # Batch loop
            if len(np.unique(samples[:, b])) != len(samples[:, b]):
                samples[:, b] = generator(size=samples[:, b].shape)
            if np.any(np.isin(samples[:, b], excludes[:, b])):
                samples[:, b] = generator(size=samples[:, b].shape)

        # check all samples are unique. Recurse if not. Because dupes are rare
        # this is efficient.
        for b in range(samples.shape[1]):  # Batch loop
            if len(np.unique(samples[:, b])) != len(samples[:, b]):
                self._remove_dupe_negs(generator, samples, excludes)
            elif np.any(np.isin(samples[:, b], excludes[:, b])):
                self._remove_dupe_negs(generator, samples, excludes)
        return samples

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    Attributes:
        _window: window size for context. E.g. window=5 means 2 words on either
            side of teh target word are used, such that there are 5 words in the
            window including the target word.
        _embed_dim: number of dimension for each word embedding.
        _batch_size: size of the batche in training.
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
                'epoch', 'batch_iters', 'iters', 'time (min)', 'loss',
                'loss(norm)', 'alpha'
            ]).astype({
                'epoch': 'int', 'batch_iters': 'int', 'iters': 'int',
                'time (min)': 'float64', 'loss': 'float64',
                'loss(norm)': 'float64', 'alpha': 'float32'
            })
        self.loss = []  # history of loss
        self.epoch = 0

        self._window = window
        self._embed_dim = embed_dim
        self._batch_size = batch_size
        self.vocab = vocab
        # State variables, used in backprop.
        self.state = {
            'context': np.empty((window - 1, batch_size), dtype=np.dtype('<U20')),
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
        context_ohe = self.vocab.encode_ohe_fast(context) / (context.shape[0])  # [V, B]
        projection = self.params['w1'] @ context_ohe  # [D,V] @ [V,B] = [D,B]
        logits = self.params['w2'] @ projection  # [V, D] x [D, B] = [V, B]
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
        self.grads['w1'][:] = 0
        self.grads['w2'][:] = 0

        dlogits = self._calc_dlogits()
        self.grads['w2'] = dlogits @ self.state['projection'].T  # [V,B] @ [B,D] = [V,D]
        dproj = self.params['w2'].T @ dlogits  # [D,V] @ [V,B] = [D,B]
        self.grads['w1'] = dproj @ self.state['context_ohe'].T  # [D,V] = [D,B] @ [B,V]

    def loss_fn(self, actual: npt.NDArray['str']) -> float:
        """Loss function differs for CBOW and S-gram

        Cbow loss is based on the single central target word.
        S-gram loss is based on the multiple surrounding context words.
        """
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
        """Equivalent to `forward`, but quicker.

        Subsets matrix columns corresponding to input context so that a smaller
        matrix is used. About 25% faster when batch=512.

        Input dim: [N-1, B]."""
        context_idx = self.vocab.encode_idx_fast(context)  # [N-1, B]
        projection_sub = self.params['w1'][:, context_idx]  # [D, V] -> [D, N-1, B]
        # CBOW averages projection of all context words.
        projection = projection_sub.mean(axis=1, keepdims=True).squeeze(axis=1)  # [D, N-1, B] -> [D, B]
        logits = self.params['w2'] @ projection  # [V, D] x [D, B] = [V, B]
        probs = softmax(logits)  # [V, B]
        # Save the forward pass state.
        self.state['context'] = context
        self.state['context_ohe'] = self.vocab.encode_ohe_fast(context)  # used by self.backward
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
        # compress the OHE array to only include positive values. This will
        # mostly be ones, but in some cases 2 and 3 are present for dupes, which
        # is why its necessary.
        context_reduced = np.take_along_axis(
            self.state['context_ohe'], context_idx, axis=0)  # [N-1, B]
        self.grads['w1'] = np.zeros_like(self.grads['w1'])
        for b in range(self.batch_size):
            self.grads['w1'][:, context_idx[:, b]] += dproj[:, [b]] @ context_reduced[:, [b]].T / context_idx.shape[0]

    def backward_quickest(self):
        """Equivalent to backward().

        Speeds up by about 25% for batch_size=512 by operating on relevant
        columns only.
        """
        dlogits = self._calc_dlogits()
        self.grads['w2'] = dlogits @ self.state['projection'].T  # [V,B] @ [B,D] = [V,D]
        dproj = self.params['w2'].T @ dlogits  # [D,V] @ [V,B] = [D,B]

        self.grads['w1'][:] = 0

        # Insert dproj into the columns of context words.
        context_idx = self.vocab.encode_idx_fast(self.state['context'])  # [N-1, B]
        x = np.arange(self.embed_dim)
        shp = (context_idx.shape[0], self.embed_dim, self.batch_size)
        rows = np.broadcast_to(x[:, np.newaxis], shp)
        cols = np.broadcast_to(context_idx[:, np.newaxis, :], shp)
        np.add.at(self.grads['w1'], (rows, cols), dproj / context_idx.shape[0])

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

    def loss_fn(self, actual: npt.NDArray):
        """Input dim for Sgram: [N-1, B]"""
        # self.state['target_str'] = actual
        # Note, target here means the words in the output layer. In the case of
        # S-gram, these are context words. In the case of CBOW, the output layer
        # is the target word. We use the CBOW terminology to remain consistent
        # with the methods inherited from Cbow.
        ohe_target = []
        for i, c in enumerate(actual):  # enumerate across context words (N-1)
            # Looping through OHE encoding ensures each OHE only has a single 1.
            ohe_target.append(self.vocab.encode_ohe_fast_single_word(np.expand_dims(c, axis=0)))
        self.state['target_ohe'] = np.stack(ohe_target, axis=2)
        self.state['loss'] = self.cross_entropy()
        return self.state['loss']

    def cross_entropy(self):
        """Cross entropy loss differs for S-gram and CBOW."""
        # Sgram has an extra dim in the output as each context word in output
        # has its own OHE.
        # dims: probs: [V, B], target_ohe: [V, B, N - 1]
        # sum across vocab, mean across context words and batch.
        return -(np.log(self.state['probs'] + 1e-9)[:, :, np.newaxis] *
                 self.state['target_ohe']).sum(0).mean()


class CbowNS(Cbow):
    """CBOW with negative sampling."""

    def forward_neg(self, target: npt.NDArray[str], context: npt.NDArray[str],
                    neg_words: npt.NDArray[str]) -> npt.NDArray[float]:
        """Similar to CBOW forward, but outputs probs for negative sampling.

        This method is not efficient and doesn't take advantage of the sparseness
        of negative sampling. It is simpler and used to validate the more
        efficient methods below.

        Differences from Cbow forward are outlined below:
            * Identical upto logits.
            * non-context and non-negative words are zeroed out in logits.
            * probs uses sigmoid rather than softmax.

        Note, this method outputs a sparse OHE probs array of dim [V, B]. Faster
        methods will generate dense [K+1, B] arrays.
        """
        context_ohe = self.vocab.encode_ohe_fast(context) / (context.shape[0])  # [V, B]
        projection = self.params['w1'] @ context_ohe  # [D,V] @ [V,B] = [D,B]
        logits = self.params['w2'] @ projection  # [V, D] x [D, B] = [V, B]

        # Zero out words that are not in context or negative samples.
        target_idx = self.vocab.encode_idx_fast(target)  # [1, B]
        neg_idx = self.vocab.encode_idx_fast(neg_words)  # [K, B]
        output_idx = np.concatenate([target_idx, neg_idx])  # [K + 1, B]
        # set non-pos and non-neg words to Inf.
        # sigmoid will make these probs=1, s.t. the loss is log(1) = 0 for these.
        # added benefit of automatically zeroing out gradients, e.g. prob-1 = 0.
        mask = np.zeros_like(logits, dtype=bool)
        for b in range(self.batch_size):
            mask[output_idx[:, b], b] = True
        logits[~mask] = np.Inf
        # negate the logits for negative words
        for b in range(self.batch_size):
            logits[neg_idx[:, b], b] *= -1

        probs = sigmoid(logits)  # [V, B], [K + N -1, B] are non-zero.
        # Save the forward_neg pass state.
        self.state['context'] = context
        self.state['target'] = target
        self.state['neg_words'] = neg_words
        self.state['context_ohe'] = context_ohe  # used by self.backward
        self.state['projection'] = projection  # used by self.backward
        self.state['probs'] = probs  # used by self.backward & self.loss_fn_neg
        return probs  # [V, B],

    def _calc_dlogits(self):
        """Similar to Cbow dlogits, but generates grads for negative sampling.

        dlogits is the only difference between regular and negative sampling
        backwards calculations. This enables us to use the inherited
        backwards method, although it would not be optimised to take
        advantage of negative samplings sparseness

        Differences from Cbow dlogits is outlined below:
            * dlogits is (p-1) for pos words. (1-p) for neg words.
                For regular CBOW/Sgram its  dlogits = p - OHE
            * since logits for non-pos and non-neg words is set to Inf, they
                automatically reduce to a zero gradient!
        """
        # dlogits for non-pos and non-neg words is 0, since they artificially
        # have prob=1.
        output_ohe = self.vocab.encode_ohe_fast(np.concatenate(
            [self.state['target'], self.state['neg_words']], axis=0))
        dlogits = (self.state['probs'] - 1)  # [V,B] - [V,B] = [V,B]

        # for dupe words we had Loss=prob**output_ohe. So differentiate by
        # multiplying by output_ohe
        dlogits *= output_ohe

        # negate the negative words, since they should be 1-prob.
        neg_idx = self.vocab.encode_idx_fast(self.state['neg_words'])  # [K, B]
        for b in range(self.batch_size):
            dlogits[neg_idx[:, b], b] *= -1
        return dlogits

    def loss_fn_neg(self) -> float:
        """Negative sampling loss function.

        loss = -log(prob_pos) - sum(log(-prob_neg))
        Since:  sigmoid(x) = 1 - sigmoid(-x)
        loss = -log(prob_pos) + sum(log(prob_neg)) - 1

        I.e. Want to maximize prob_pos, and minimize prob_neg) to reduce loss.
        """
        # sum across pos and neg words, mean across batches.
        # because we applied -1 to the negative words in the forward_neg pass we
        # can simply take the log and sum all probs

        # For duplicate words (in the output layer), we need to apply p**counts.
        # This is the same as repeating the log(p) term in the loss function for
        # each occurrence of the word.
        if self.state['probs'].shape == (self.vocab.size, self.batch_size):
            output_ohe = self.vocab.encode_ohe_fast(np.concatenate(
                [self.state['target'], self.state['neg_words']], axis=0))
            probs = self.state['probs'] ** output_ohe
        else:
            probs = self.state['probs']
        self.state['loss'] = -np.log(probs).sum(0).mean()  # [K+1, B] -> scalar
        return self.state['loss']

    def forward_neg_quick(self, target, context, neg_words):
        """Similar to forward_neg, operates only on activated words and uses
        dense arrays.

        Input dims:
            target: (a.k.a. output layer): S-gram[N-1, B], CBOW[1, B]
            context: (a.k.a. input layer): S-gram[1, B], CBOW[N-1, B]
            neg_words: [K, B]
        Output dims:
            probs: [K+1, B]. Vs forward_neg [V, B]
         """
        # Operating under assumption that neg words don't intersect with target
        # or context words.
        output = np.vstack([target, neg_words])  # [1+K, B]
        # neg_indicator applies -1 to neg words on the logit values.
        neg_mask = np.array([False] * target.shape[0] + [True] * neg_words.shape[0])
        neg_indicator = np.ones(output.shape[0])  # [1+k]
        neg_indicator[neg_mask] = -1  # set neg words to -1.

        input_idx = self.vocab.encode_idx_fast(context)  # [N-1, B]
        output_idx = self.vocab.encode_idx_fast(output)  # [1+K, B]

        projection_sub = self.params['w1'][:, input_idx]  # [D, N-1, B]
        projection = projection_sub.mean(axis=1, keepdims=True).squeeze(axis=1)  # [D, N-1, B] -> [D, B]

        # logits includes both pos and neg words. Sum over embed dim, so we have
        # a logit for each word (pos and neg) and each batch.
        logits = (self.params['w2'][output_idx, :] * projection.T).sum(-1)  # [1+K, B, D] * [D, B].T).sum(-1) = [1+K, B]
        # Negative words are negated inside the sigmoid in the loss functions.
        logits *= neg_indicator[:, np.newaxis]

        probs = sigmoid(logits)  # [1+K, B]
        # Save the forward_neg pass state.
        self.state['context'] = context
        self.state['target'] = target
        self.state['neg_words'] = neg_words
        self.state['input_idx'] = input_idx  # used in self.backward_neg_quick
        self.state['output_idx'] = output_idx  # used in self.backward_neg_quick
        self.state['neg_mask'] = neg_mask
        self.state['probs'] = probs
        self.state['projection'] = projection
        return probs  # [K+1, B]

    def backward_neg(self):
        # backward method is similar for regular and negative sampling.
        # The only difference is captured by _calc_dlogits()
        # Only works with forward_neg.
        self.backward()

    def backward_neg_quick(self):
        """Same as backward_neg, except only operates on pos and neg words."""
        # Only works with forward_neg_quick
        input_idx = self.state['input_idx']  # [N-1, B]
        probs_pos = self.state['probs'][~self.state['neg_mask'], :]  # [1, B]
        probs_neg = self.state['probs'][self.state['neg_mask'], :]  # [K, B]
        pos_idx = self.state['output_idx'][~self.state['neg_mask']]  # [1, B]
        neg_idx = self.state['output_idx'][self.state['neg_mask']]  # [K, B]

        # Input Matrix Gradients
        dh_pos = np.zeros((self.embed_dim, self.batch_size))
        dh_neg = np.zeros((self.embed_dim, self.batch_size))
        for b in range(self.batch_size):
            dh_pos[:, b] = ((probs_pos - 1)[:, b, np.newaxis] *
                            self.params['w2'][pos_idx[:, b], :]).sum(0)  # CBow: [1, 1] * [D, 1]  = [1, D]. Sgram: [N-1, 1] * [D, 1] = [N-1,D].sum(0)
            dh_neg[:, b] = ((1 - probs_neg)[:, b, np.newaxis] *
                            self.params['w2'][neg_idx[:, b], :]).sum(0)  # [K, 1] * [K, D]  = [K, D].sum(0) = [1, D]
        dh = dh_pos + dh_neg  # [D, B]
        self.grads['w1'][:] = 0
        # mean acros batches. Distribute gradient across each input words vector.
        for b in range(self.batch_size):
            # Need to loop over each context word sing += only does single
            # operation for duplicate words
            for i in range(input_idx.shape[0]):
                self.grads['w1'][:, input_idx[i, b]] += dh[:, b] / input_idx.shape[0]  # [N -1, D] -> [N-1, D]

        # Output Matrix Gradients
        dw2_pos = np.zeros((self.vocab.size, self.embed_dim))
        dw2_neg = np.zeros((self.vocab.size, self.embed_dim))
        for b in range(self.batch_size):
            dw2_neg[neg_idx[:, b], :] += ((1 - probs_neg[:, b, np.newaxis]) *
                                          self.state['projection'][:, b])  # [K, 1] * [D, 1]  = [D, K]
            for i in range(pos_idx.shape[0]):  # pos words can be duplicated. So need add each one in a loop.
                dw2_pos[pos_idx[i, b], :] += ((probs_pos[i, b, np.newaxis] - 1) *
                                              self.state['projection'][:, b])  # [1, 1] * [D, 1]  = [D, 1]
        self.grads['w2'] = (dw2_pos + dw2_neg)

    def backward_neg_quickest(self):
        """Same as backward_neg_quick, except uses optimized np indexing.

        It's not easy to understand the matrix operations in this method, so
        slower more explicit methods are retained for pedagogical reasons.
        """
        input_idx = self.state['input_idx']  # [N-1, B]
        probs_pos = self.state['probs'][~self.state['neg_mask'], :]  # [1, B]
        probs_neg = self.state['probs'][self.state['neg_mask'], :]  # [K, B]
        pos_idx = self.state['output_idx'][~self.state['neg_mask']]  # [1, B]
        neg_idx = self.state['output_idx'][self.state['neg_mask']]  # [K, B]

        idx = np.vstack([pos_idx, neg_idx])  # [1+K, B]
        probs = np.vstack([probs_pos - 1, 1 - probs_neg])  # [1+K, B]

        # Input Matrix Gradients
        self.grads['w1'][:] = 0
        dh = np.einsum('kb,kbd->db', probs, self.params['w2'][idx, :])  # [K+1, B], [K+1, B, D] -> [D, B]
        dh /= input_idx.shape[0]  # [D, B] Distribute gradient across each input word.
        # Insert gradient (dh) into relevant words (columns) of d_w1.
        x = np.arange(self.embed_dim)
        shp = (input_idx.shape[0], self.embed_dim, self.batch_size)
        rows = np.broadcast_to(x[:, np.newaxis], shp)
        cols = np.broadcast_to(input_idx[:, np.newaxis, :], shp)
        np.add.at(self.grads['w1'], (rows, cols), dh)

        # Output Matrix Gradients
        self.grads['w2'][:] = 0
        # einsums below have a more compressed dw2 matrix, that can be inserted
        # directly into the dw2 matrix using indexing.
        dw2 = np.einsum('kb,db->kdb', probs, self.state['projection'])  # [K+1, B],[D, B] -> [K+1, D, B]
        x = np.arange(self.embed_dim)
        shp = (pos_idx.shape[0] + neg_idx.shape[0], self.embed_dim, self.batch_size)
        cols = np.broadcast_to(x[:, np.newaxis], shp)
        rows = np.broadcast_to(idx[:, np.newaxis, :], shp)
        np.add.at(self.grads['w2'], (rows, cols), dw2)

    def loss_normalized(self):
        """Calculates the standard loss using non-negative sampling.

        This enables useful comparison with standard Cbow and Sgram loss
        curves. This is a relatively slow operation, so should not be called
        frequently.
        """
        # invoke the softmax forward method to get comparable loss.
        _ = self.forward_quick(self.state['context'])
        return self.loss_fn(self.state['target'])


class SgramNS(CbowNS, Sgram):
    def forward_neg(self,
                    target: npt.NDArray[str],
                    context: npt.NDArray[str],
                    neg_words: npt.NDArray[str]) -> npt.NDArray[float]:
        # For s-gram, we use Cbow forward method, but witch target and context.
        return super().forward_neg(
            target=context, context=target, neg_words=neg_words)

    def forward_neg_quick(self,
                          target: npt.NDArray[str],
                          context: npt.NDArray[str],
                          neg_words: npt.NDArray[str]) -> npt.NDArray[float]:
        # For s-gram, we use Cbow forward method, but witch target and context.
        return super().forward_neg_quick(
            target=context, context=target, neg_words=neg_words)

    def loss_normalized(self):
        """Calculates the standard loss using non-negative sampling.

        This enables useful comparison with standard Cbow and Sgram loss
        curves. This is a relatively slow operation, so should not be called
        frequently.
        """
        # invoke the softmax forward method to get comparable loss.
        _ = self.forward_quick(self.state['context'])
        return Sgram.loss_fn(self, self.state['target'])
