import timeit
import numpy as np
import pickle
from pathlib import Path

import w2v_numpy


if __name__ == '__main__':
    """Script tests speed of methods and checks equivalence of regular and
    optimized methods."""
    VECTOR_DIM = 300
    BASE_DIR = Path(__file__).parent
    DATA_FILE = BASE_DIR / 'data/toy_data_41mb_raw.txt.train'
    # LOAD_MODEL = True
    epochs = 10
    batch = 512
    window = 7

    with open(DATA_FILE, 'rb') as fp:
        train_data = pickle.load(fp)

    vocab = w2v_numpy.Vocab(train_data)
    cbow = w2v_numpy.Cbow(vocab, VECTOR_DIM, window, batch, 1)
    sgram = w2v_numpy.Sgram(vocab, VECTOR_DIM, window, batch, 1)
    data = w2v_numpy.Dataloader(train_data, cbow.window)

    batched_data = w2v_numpy.grouper(data, batch)
    chunk = next(batched_data)
    c, t = zip(*chunk)
    context = np.array(c).T
    target = np.expand_dims(np.array(t), 0)
    cbow = w2v_numpy.Cbow(vocab, VECTOR_DIM, window, batch, seed=1)

    # Requires context and vocab
    print('\nTesting Encoding Speed from token to OHE vector: Vocab.encod_ohe...')
    tmp = timeit.Timer('vocab.encode_ohe_ignore_dupe(context)', globals=globals()).repeat(1, 10)
    print(f'encode_ohe_ignore_dupe:{tmp[0]/10:.3}s per run')
    tmp = timeit.Timer('vocab.encode_ohe(context)', globals=globals()).repeat(1, 10)
    print(f'encode_ohe:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('vocab.encode_ohe_fast(context)', globals=globals()).repeat(1, 100)
    print(f'encode_ohe_fast:{tmp[0] / 100:.3}s per run')
    print('\n')

    print('Testing Encoding from token to index number: Vocab.encod_idx...')
    tmp = timeit.Timer('vocab.encode_idx(context)', globals=globals()).repeat(1, 10)
    print(f'encode_idx:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('vocab.encode_idx_fast(context)', globals=globals()).repeat(1, 100)
    print(f'encode_idx_fast:{tmp[0] / 100:.3}s per run')
    print('\n')

    print('Testing equality of output of encoding algos')
    tmp = np.array_equal(vocab.encode_ohe(context), vocab.encode_ohe_fast(context))
    print(f'{'passed' if tmp else 'failed'}: encode_ohe == encode_ohe_fast')
    tmp = np.array_equal(vocab.encode_idx(context), vocab.encode_idx_fast(context))
    print(f'{'passed' if tmp else 'failed'}: encode_idx == encode_idx_fast')
    tmp = np.array_equal(vocab.encode_ohe_ignore_dupe(context), vocab.encode_ohe_fast(context))
    print(f'{'passed' if not tmp else 'failed'}: encode_ohe_ignore_dupe != encode_ohe')
    print('\n')

    # test dataloader
    print('Testing Dataloader speed:')
    test_data = w2v_numpy.grouper(w2v_numpy.Dataloader(train_data, cbow.window), batch)
    tmp = timeit.Timer('next(test_data)', globals=globals()).repeat(1, 1000)
    print(f'Dataloader iteration:{tmp[0] / 1000:.3}s per iter')

    # CBOW tests
    # test CBOW forward speed. Requires context.
    print('\nCBOW - Testing forward, backward, loss and opt speed:')
    cbow = w2v_numpy.Cbow(vocab, VECTOR_DIM, window, batch, seed=1)
    tmp = timeit.Timer('cbow.forward(context)', globals=globals()).repeat(1, 10)
    print(f'forward:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('cbow.forward_quick(context)', globals=globals()).repeat(1, 10)
    print(f'forward_quick:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('cbow.loss_fn(target)', globals=globals()).repeat(1, 10)
    print(f'loss_fn:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('cbow.backward()', globals=globals()).repeat(1, 10)
    print(f'backward:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('cbow.backward_quick()', globals=globals()).repeat(1, 10)
    print(f'backward_quick:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('cbow.optim_sgd(0.00001)', globals=globals()).repeat(1, 10)
    print(f'optim_sgd:{tmp[0] / 10:.3}s per run')
    print('\n')

    print('Testing equality of output of forward and backward algos')
    tmp = np.allclose(cbow.forward(context), cbow.forward_quick(context))
    print(f'{'passed' if tmp else 'failed'}: cbow.forward_quick == cbow.forward ')

    # create w1 and w1 gradients
    cbow.forward(context)
    cbow.loss_fn(target)
    cbow.backward_quick()
    grads_quick = cbow.grads
    cbow.backward()
    grads = cbow.grads
    tmp = np.allclose(grads_quick['w1'], grads['w1'])
    print(f'{'passed' if tmp else 'failed'}: cbow.backward_quick dw1 == cbow.backward dw1')
    tmp = np.allclose(grads_quick['w2'], grads['w2'])
    print(f'{'passed' if tmp else 'failed'}: cbow.backward_quick dw2 == cbow.backward dw2')


    # S-gram tests
    # tests S-gram forward speed. Requires context.
    print('\nS-gram - Testing forward, backward, loss and opt speed:')
    sgram = w2v_numpy.Sgram(vocab, VECTOR_DIM, window, batch, seed=1)
    tmp = timeit.Timer('sgram.forward(target)', globals=globals()).repeat(1, 10)
    print(f'forward:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('sgram.forward_quick(target)', globals=globals()).repeat(1, 10)
    print(f'forward_quick:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('sgram.loss_fn(context)', globals=globals()).repeat(1, 10)
    print(f'loss_fn:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('sgram.backward()', globals=globals()).repeat(1, 10)
    print(f'backward:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('sgram.backward_quick()', globals=globals()).repeat(1, 10)
    print(f'backward_quick:{tmp[0] / 10:.3}s per run')
    tmp = timeit.Timer('sgram.optim_sgd(0.00001)', globals=globals()).repeat(1, 10)
    print(f'optim_sgd:{tmp[0] / 10:.3}s per run')
    print('\n')

    print('Testing equality of output of forward and backward algos')
    tmp = np.allclose(sgram.forward(context), sgram.forward_quick(context))
    print(f'{'passed' if tmp else 'failed'}: sgram.forward_quick == sgram.forward ')

    # create w1 and w1 gradients
    sgram.forward(context)
    sgram.loss_fn(target)
    sgram.backward_quick()
    grads_quick = sgram.grads
    sgram.backward()
    grads = sgram.grads
    tmp = np.allclose(grads_quick['w1'], grads['w1'])
    print(f'{'passed' if tmp else 'failed'}: sgram.backward_quick dw1 == sgram.backward dw1')
    tmp = np.allclose(grads_quick['w2'], grads['w2'])
    print(f'{'passed' if tmp else 'failed'}: sgram.backward_quick dw2 == sgram.backward dw2')