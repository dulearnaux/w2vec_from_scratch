from w2v_numpy import *

import matplotlib
import pickle
from pathlib import Path
import argparse
import sys
import time

# Input args:
#  -d= path to working dir
#  -s, include to start new model, or exclude to load existing.
#  -w= window length to use. must be odd and >= 3.
#  -v= embedding vector length to train.
#  -b= batch size.
#  -e= number of epochs to train. Default set to 1000, which is too large.
#      user is expected to manually interrupt training when results are
#      sufficient.

parser = argparse.ArgumentParser(
    description="""      
        Run the Continuous Bag of Words model from the original word2vec paper.
        Efficient Estimation of Word Representations in Vector Space.
        https://arxiv.org/pdf/1301.3781"""
)
parser.add_argument('-d', '--dir', nargs='?',  type=Path,
                    default=Path(__file__).parent, dest='base_dir',
                    help=('base directory to save/load files to/from. Should '
                          'contain existing sub dirs and file: '
                          'checkpoint, data/toy_data_41mb_raw.txt.train'))

parser.add_argument('-s', '--start_new_model',
                    action=argparse.BooleanOptionalAction,
                    help='Will start a new model from scratch if passed. '
                         'Otherwise looks for cbow.model in dir to load it.')
parser.add_argument('-w', '--window', nargs='?', default=7, type=int,
                    help='window size for cbow model. E.g. for window=5, 2 words'
                         ' on the left and 2 words on the right of the target'
                         ' word are used as context.')
parser.add_argument('-v', '--vector_dim', nargs='?', default=300, type=int,
                    help='number of embedding dimensions to use for the cbow')
parser.add_argument('-b', '--batch_size', nargs='?', default=512, type=int)
parser.add_argument('-e', '--epochs', nargs='?', default=1000, type=int)

args = parser.parse_args()

DATA_FILE = args.base_dir / 'data/toy_data_41mb_raw.txt.train'


# Wrappers to streamline code in the training loop.
def train_batch():
    """Wrapper around training steps."""
    c, t = zip(*chunk)
    context = np.array(c).T
    target = np.expand_dims(np.array(t), 0)

    cbow.forward_quick(context)
    cbow.loss.append(cbow.loss_fn(target))
    cbow.backward_quick()
    cbow.optim_sgd(alpha)

def stats_print():
    """Prints stats to screen and saves them to model obj."""
    print(f'epoch: {cbow.epoch},   '
          f'line_no {data.line_no:05} of {len(data.data)},   '
          f'time: {(time.time() - start) / 60:.6}min,   '
          f'loss:{cbow.loss[-1]:.5}    '
          )
    # Keep training progress for later analysis of hyperparams.
    cbow.stats.loc[len(cbow.stats)] = [
        cbow.epoch, batch_counter, batch_counter * args.batch_size,
                                   (time.time() - start) / 60, cbow.loss[-1], alpha]

def stats_save():
    """Saves model and plots to files."""
    cbow.plot_loss_curve(args.base_dir / 'loss.png')
    cbow.plot_prediction(data, args.base_dir / 'pred.png')
    with open(args.base_dir / 'cbow.model', 'wb') as fp:
        pickle.dump({'model': cbow}, fp)

def load_model():
    if args.start_new_model:
        cbow = Cbow(vocab, args.vector_dim, args.window, args.batch_size, seed=1)
    else:
        try:
            with open(args.base_dir / 'cbow.model', 'rb') as fp:
                tmp = pickle.load(fp)
            cbow = tmp['model']
            cbow._batch_size = args.batch_size
            cbow._skip_gram = False  # added skip gram after some cbow models were pickled.
        except FileNotFoundError:
            print('cbow.model does not exist. Check directory, '
                  'or use start_new_model arg to start from scratch')
            sys.exit(1)
    return cbow


if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    print(f'Working dir: {args.base_dir}\n')
    print(f'{args.base_dir / DATA_FILE =}')

    # train data is a list of news stories with variable number of tokens per story.
    # Each news story has fully processed tokens only. Tokens across stories are not
    # related, so sampling of a context window has to be within a story.
    with open(DATA_FILE, 'rb') as fp:
        train_data = pickle.load(fp)

    vocab = Vocab(train_data)
    cbow = load_model()
    window = cbow.window  # overwrite if loaded model is different.
    start = time.time()
    epoch = cbow.epoch
    for epoch in range(epoch - 1, args.epochs):
        cbow.epoch += 1
        batch_counter = 0
        data = Dataloader(train_data, cbow.window)
        batched_data = grouper(data, args.batch_size)

        for chunk in batched_data:
            batch_counter += 1
            alpha = 0.007 if epoch < 70 else 0.0008
            train_batch()

            if batch_counter % 10 == 0:
                stats_save()
            if batch_counter % 100 == 0 or data.line_no == 1:
                stats_print()

        # Save checkpoint at the end of each epoch. Make sure disk has enough
        # space. Might need to delete intermediate checkpoints as you go.
        with open(args.base_dir / 'checkpoints' / f'cbow.model.{cbow.epoch:04}',
                  'wb') as fp:
            pickle.dump({'model': cbow}, fp)

    # realistically, the epoch loop isn't completed. Progress is tracked and
    # the program will get manually terminated long before all epochs are run.
    end = time.time()
    with open(args.base_dir / 'cbow.model', 'wb') as fp:
        pickle.dump({'model': cbow}, fp)
