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
        Run the Skipgram model from the original word2vec paper.
        Efficient Estimation of Word Representations in Vector Space.
        https://arxiv.org/pdf/1301.3781.
        
        Note, skip gram is just the reverse of teh CBOW model. The target is fed 
        as the input OHE. The loss is calculated against each context OHE
        vector. So it reuses most of the code from teh CBOW model. 
        """
)
parser.add_argument('-d', '--dir', nargs='?',  type=Path,
                    default=Path(__file__).parent, dest='base_dir',
                    help=('base directory to save/load files to/from. Should '
                          'contain existing sub dirs and file: '
                          'checkpoint, data/toy_data_41mb_raw.txt.train'))

parser.add_argument('-s', '--start_new_model',
                    action=argparse.BooleanOptionalAction,
                    help='Will start a new model from scratch if passed. '
                         'Otherwise looks for sgram.model in dir to load it.')
parser.add_argument('-w', '--window', nargs='?', default=5, type=int,
                    help='window size for sgram model. E.g. for window=5, 2 words'
                         ' on the left and 2 words on the right of the target'
                         ' word are used as context.')
parser.add_argument('-v', '--vector_dim', nargs='?', default=300, type=int,
                    help='number of embedding dimensions to use for the sgram')
parser.add_argument('-b', '--batch_size', nargs='?', default=512, type=int)
parser.add_argument('-e', '--epochs', nargs='?', default=1000, type=int)

args = parser.parse_args()

DATA_FILE = args.base_dir / 'data/toy_data_41mb_raw.txt.train'


# Wrappers to streamline code in the training loop.
def train_sgram_batch():
    """Wrapper around training steps."""
    c, t = zip(*chunk)
    context = np.array(c).T
    target = np.expand_dims(np.array(t), 0)

    # Note, target and context are switched for skip gram.
    sgram.forward_quick(target)
    sgram.loss.append(sgram.loss_fn_sgram(context))
    sgram.backward()
    # sgram.backward_quick()
    sgram.optim_sgd(alpha)


def stats_print():
    """Prints stats to screen and saves them to model obj."""
    print(f'epoch: {sgram.epoch},   '
          f'line_no {data.line_no:05} of {len(data.data)},   '
          f'time: {(time.time() - start) / 60:.6}min,   '
          f'loss:{sgram.loss[-1]:.5}    '
          )
    # Keep training progress for later analysis of hyperparams.
    sgram.stats.loc[len(sgram.stats)] = [
        sgram.epoch, batch_counter, batch_counter * args.batch_size,
                                    (time.time() - start) / 60, sgram.loss[-1], alpha]


def stats_save():
    """Saves model and plots to files."""
    sgram.plot_loss_curve(args.base_dir / 'sgram.loss.png')
    sgram.plot_prediction(data, args.base_dir / 'sgram.pred.png')
    with open(args.base_dir / 'sgram.model', 'wb') as fp:
        pickle.dump({'model': sgram}, fp)


def load_model():
    if args.start_new_model:
        sgram = Cbow(vocab, args.vector_dim, args.window, args.batch_size,
                     seed=1, skip_gram=True)
    else:
        try:
            with open(args.base_dir / 'sgram.model', 'rb') as fp:
                tmp = pickle.load(fp)
            sgram = tmp['model']
            sgram._batch_size = args.batch_size
        except FileNotFoundError:
            print('sgram.model does not exist. Check directory, '
                  'or use start_new_model arg to start from scratch')
            sys.exit(1)
    return sgram



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
    sgram = load_model()
    window = sgram.window  # overwrite if loaded model is different.
    start = time.time()
    epoch = sgram.epoch
    for epoch in range(epoch - 1, args.epochs):
        sgram.epoch += 1
        batch_counter = 0
        data = Dataloader(train_data, sgram.window)
        batched_data = grouper(data, args.batch_size)

        for chunk in batched_data:
            batch_counter += 1
            alpha = 0.08 if epoch < 10 else 0.0005
            train_sgram_batch()

            if batch_counter % 10 == 0:
                stats_save()
            if batch_counter % 100 == 0 or data.line_no == 1:
                stats_print()

        # Save checkpoint at the end of each epoch. Make sure disk has enough
        # space. Might need to delete intermediate checkpoints as you go.
        with open(args.base_dir / 'checkpoints' / f'sgram.model.{sgram.epoch:04}',
                  'wb') as fp:
            pickle.dump({'model': sgram}, fp)

    # realistically, the epoch loop isn't completed. Progress is tracked and
    # the program will get manually terminated long before all epochs are run.
    end = time.time()
    with open(args.base_dir / 'sgram.model', 'wb') as fp:
        pickle.dump({'model': sgram}, fp)
