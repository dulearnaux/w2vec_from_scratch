import copy
from unittest import TestCase

from pathlib import Path
import pickle
import w2v_numpy as w2v
import numpy as np


class Setup:
    """Loads and generates data common to multiple test classes."""

    @classmethod
    def setup(cls):
        data_file = Path(__file__).parent / 'data/toy_data_41mb_raw.txt.train'
        with open(data_file, 'rb') as fp:
            cls.train_data = pickle.load(fp)

        cls.alpha = 0.005
        cls.batch_size = 512
        cls.vector_dim = 300
        cls.window = 7

        cls.vocab = w2v.Vocab(cls.train_data)
        cls.data = w2v.Dataloader(cls.train_data, cls.window)
        cls.batched_data = w2v.grouper(cls.data, cls.batch_size)
        for i in range(22):  # choose a random batch number to use
            chunk = next(cls.batched_data)
        c, t = zip(*chunk)
        cls.context = np.array(c).T  # [N-1, B]
        cls.target = np.expand_dims(np.array(t), 0)  # [1, B]


class TestVocab(TestCase):

    @classmethod
    def setUpClass(cls):
        Setup.setup()

    def test_encode_ohe_fast(self):
        encode_ohe = Setup.vocab.encode_ohe(Setup.target)
        encode_ohe_fast = Setup.vocab.encode_ohe_fast(Setup.target)
        self.assertTrue(np.array_equal(encode_ohe, encode_ohe_fast))

        encode_ohe = Setup.vocab.encode_ohe(Setup.context)
        encode_ohe_fast = Setup.vocab.encode_ohe_fast(Setup.context)
        self.assertTrue(np.array_equal(encode_ohe, encode_ohe_fast))

    def test_encode_idx_fast(self):
        encode_idx = Setup.vocab.encode_idx(Setup.target)
        encode_idx_fast = Setup.vocab.encode_idx_fast(Setup.target)
        self.assertTrue(np.array_equal(encode_idx, encode_idx_fast))

        encode_idx = Setup.vocab.encode_idx(Setup.context)
        encode_idx_fast = Setup.vocab.encode_idx_fast(Setup.context)
        self.assertTrue(np.array_equal(encode_idx, encode_idx_fast))

    def test_encode_ohe_fast_single_word(self):
        encode_ohe = Setup.vocab.encode_ohe(Setup.target)
        encode_ohe_fast_single_word = Setup.vocab.encode_ohe_fast_single_word(
            Setup.target)
        self.assertTrue(np.array_equal(encode_ohe, encode_ohe_fast_single_word))

        with self.assertRaises(AssertionError):
            Setup.vocab.encode_ohe_fast_single_word(Setup.context)


class TestCbow(TestCase):

    @classmethod
    def setUpClass(cls):
        Setup.setup()
        cls.cbow = w2v.Cbow(
            Setup.vocab, Setup.vector_dim, Setup.window, Setup.batch_size,
            seed=1)

    def test_cbow_flow(self):
        try:
            _ = self.cbow.forward_quick(Setup.context)
            _ = self.cbow.loss_fn(Setup.target)
            self.cbow.backward()
            self.cbow.optim_sgd(Setup.alpha)
        except Exception:
            self.fail(f'Cbow failed training workflow test.')

    def test_forward_quick_cbow(self):
        # two forward methods should give the same probs.
        probs_fwd = self.cbow.forward(Setup.context)
        probs_quick = self.cbow.forward_quick(Setup.context)
        self.assertTrue(np.allclose(probs_fwd, probs_quick))

    def test_backward_quick(self):
        cbow_quick = copy.deepcopy(self.cbow)

        self.cbow.forward(Setup.context)
        _ = self.cbow.loss_fn(Setup.target)
        self.cbow.backward()

        cbow_quick.forward_quick(Setup.context)
        cbow_quick.loss_fn(Setup.target)
        cbow_quick.backward_quick()
        self.assertTrue(np.allclose(self.cbow.grads['w1'], cbow_quick.grads['w1']))
        self.assertTrue(np.allclose(self.cbow.grads['w2'], cbow_quick.grads['w2']))

    def test_backward_quickest(self):
        cbow_test = copy.deepcopy(self.cbow)

        self.cbow.forward(Setup.context)
        _ = self.cbow.loss_fn(Setup.target)
        self.cbow.backward()

        cbow_test.forward_quick(Setup.context)
        cbow_test.loss_fn(Setup.target)
        cbow_test.backward_quickest()
        self.assertTrue(np.allclose(self.cbow.grads['w1'], cbow_test.grads['w1']))
        self.assertTrue(np.allclose(self.cbow.grads['w2'], cbow_test.grads['w2']))

    def test_no_batch(self):
        """Should pass id batch dim is preserved but len=1. Fails if batch dim
        is squeezed."""

        cbow_nb = w2v.Cbow(Setup.vocab, Setup.vector_dim, Setup.window, batch_size=1, seed=1)

        with self.assertRaises(KeyError):
            cbow_nb.forward(Setup.context[:, 0])
        try:
            _ = cbow_nb.forward(Setup.context[:, 0, np.newaxis])  # should work if dim is passed as list
        except Exception as e:
            self.fail(f'Cbow.forward failed with no batch, raised {e}')

        with self.assertRaises(KeyError):
            cbow_nb.forward_quick(Setup.context[:, 0])
        try:
            _ = cbow_nb.forward_quick(Setup.context[:, 0, np.newaxis])
        except Exception as e:
            self.fail(f'Cbow.forward_quick failed with no batch, raised {e}')

        with self.assertRaises(KeyError):
            cbow_nb.loss_fn(Setup.target[:, 0])
        try:
            _ = cbow_nb.loss_fn(Setup.target[:, 0, np.newaxis])
        except Exception as e:
            self.fail(f'Cbow.loss_fn failed with no batch, raised {e}')

        try:
            cbow_nb.backward()
        except Exception as e:
            self.fail(f'Cbow.backward failed with no batch, raised {e}')

        try:
            cbow_nb.backward_quick()
        except Exception as e:
            self.fail(f'Cbow.backward_quick failed with no batch, raised {e}')

        try:
            cbow_nb.backward_quickest()
        except Exception as e:
            self.fail(f'Cbow.backward_quickest failed with no batch, raised {e}')

        try:
            cbow_nb.optim_sgd(Setup.alpha)
        except Exception as e:
            self.fail(f'Cbow.optim_sgd failed with no batch, raised {e}')


class TestSgram(TestCase):

    @classmethod
    def setUpClass(cls):
        Setup.setup()
        cls.sgram = w2v.Sgram(
            Setup.vocab, Setup.vector_dim, Setup.window, Setup.batch_size,
            seed=1)

    def test_sgram_flow(self):
        try:
            _ = self.sgram.forward_quick(Setup.target)
            _ = self.sgram.loss_fn(Setup.context)
            self.sgram.backward()
            self.sgram.optim_sgd(Setup.alpha)
        except Exception:
            self.fail(f'Sgram failed training workflow test.')

    def test_forward_quick(self):
        # two forward methods should give the same probs.
        probs_fwd = self.sgram.forward(Setup.target)
        probs_quick = self.sgram.forward_quick(Setup.target)
        self.assertTrue(np.allclose(probs_fwd, probs_quick))

    def test_backward_quick(self):
        sgram_quick = copy.deepcopy(self.sgram)

        self.sgram.forward(Setup.target)
        _ = self.sgram.loss_fn(Setup.context)
        self.sgram.backward()

        sgram_quick.forward_quick(Setup.target)
        sgram_quick.loss_fn(Setup.context)
        sgram_quick.backward_quick()
        self.assertTrue(np.allclose(self.sgram.grads['w1'], sgram_quick.grads['w1']))
        self.assertTrue(np.allclose(self.sgram.grads['w2'], sgram_quick.grads['w2']))

    def test_backward_quickest(self):
        sgram_quick = copy.deepcopy(self.sgram)

        self.sgram.forward(Setup.target)
        _ = self.sgram.loss_fn(Setup.context)
        self.sgram.backward()

        sgram_quick.forward_quick(Setup.target)
        sgram_quick.loss_fn(Setup.context)
        sgram_quick.backward_quickest()
        self.assertTrue(np.allclose(self.sgram.grads['w1'], sgram_quick.grads['w1']))
        self.assertTrue(np.allclose(self.sgram.grads['w2'], sgram_quick.grads['w2']))


class TestCbowNS(TestCase):

    @classmethod
    def setUpClass(cls):
        Setup.setup()

        k = 3
        cls.cbow_ns = w2v.CbowNS(Setup.vocab, Setup.vector_dim, Setup.window, Setup.batch_size, seed=1)
        cls.data = w2v.Dataloader(Setup.train_data, cls.cbow_ns.window, negative_samples=k)
        cls.neg_words = cls.data.neg_samples(Setup.context, Setup.target)

    def test_cbow_ns_flow(self):
        try:
            _ = self.cbow_ns.forward_neg_quick(
                Setup.target, Setup.context, self.neg_words)
            _ = self.cbow_ns.loss_fn_neg()
            self.cbow_ns.backward_neg_quickest()
            self.cbow_ns.optim_sgd(Setup.alpha)
        except Exception:
            self.fail(f'CBOW NEG failed training workflow test.')

    def test_forward_neg(self):
        # two forward_neg methods should give the same probs.
        probs_ohe = self.cbow_ns.forward_neg(
            Setup.target, Setup.context, self.neg_words)  # [V, B] OHE vectors
        probs_neg = self.cbow_ns.forward_neg_quick(
            Setup.target, Setup.context, self.neg_words)  # [K+1, B] dense probs.

        target_idx = Setup.vocab.encode_idx_fast(Setup.target)
        neg_idx = Setup.vocab.encode_idx_fast(self.neg_words)
        idx = np.concatenate([target_idx, neg_idx], axis=0)
        probs_dense = np.zeros_like(probs_neg)
        for b in range(self.cbow_ns.batch_size):
            probs_dense[:, b] += probs_ohe[idx[:, b], b]
        self.assertTrue(np.allclose(probs_neg, probs_dense))

    def test_backward_neg(self):
        mdl_neg = copy.deepcopy(self.cbow_ns)

        self.cbow_ns.forward_neg(Setup.target, Setup.context, self.neg_words)
        _ = self.cbow_ns.loss_fn_neg()
        self.cbow_ns.backward_neg()

        mdl_neg.forward_neg_quick(Setup.target, Setup.context, self.neg_words)
        mdl_neg.loss_fn_neg()
        mdl_neg.backward_neg_quick()
        self.assertTrue(np.allclose(self.cbow_ns.grads['w1'], mdl_neg.grads['w1']))
        self.assertTrue(np.allclose(self.cbow_ns.grads['w2'], mdl_neg.grads['w2']))

    def test_backward_neg_quickest(self):
        mdl_neg = copy.deepcopy(self.cbow_ns)

        self.cbow_ns.forward_neg(Setup.target, Setup.context, self.neg_words)
        _ = self.cbow_ns.loss_fn_neg()
        self.cbow_ns.backward_neg()

        mdl_neg.forward_neg_quick(Setup.target, Setup.context, self.neg_words)
        mdl_neg.loss_fn_neg()
        mdl_neg.backward_neg_quickest()
        self.assertTrue(np.allclose(self.cbow_ns.grads['w1'], mdl_neg.grads['w1']))
        self.assertTrue(np.allclose(self.cbow_ns.grads['w2'], mdl_neg.grads['w2']))

    def test_incompatibility(self):
        """Test CbowNS.backward() following CbowNS().forward_neg_quick()"""
        self.cbow_ns.forward_neg(
            Setup.target, Setup.context, self.neg_words)
        _ = self.cbow_ns.loss_fn_neg()
        with self.assertRaises((KeyError, ValueError, IndexError)):
            self.cbow_ns.backward_neg_quick()

        self.cbow_ns.forward_neg_quick(
            Setup.target, Setup.context, self.neg_words)
        _ = self.cbow_ns.loss_fn_neg()
        with self.assertRaises((IndexError, ValueError)):
            self.cbow_ns.backward()


