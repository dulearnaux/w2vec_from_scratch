from typing import Union

from w2v_numpy import *

import pickle
from pathlib import Path
import argparse
import sys
import time

# Input args:
#  -b= path to base directory
#  -m= model filename. Relative to base directory.
#  -d= Training data filename. Relative to base directory.
#  -o, include to overwrite existing model or start new model. Exclude to load
#       existing.
#  -t= ['cbow', 'sgram']. Use CBOW or SGRAM model. Defaults to CBOW.
#  -w= window length to use. must be odd and >= 3.
#  -v= embedding vector length to train.
#  -B= batch size.
#  -e= number of epochs to train. Default set to 1000, which is too large.
#      user is expected to manually interrupt training when results are
#      sufficient.
#  -a= alpha, learning rate. Default is 0.05,

parser = argparse.ArgumentParser(
    description="""      
        Run the Continuous Bag of Words model from the original word2vec paper.
        Efficient Estimation of Word Representations in Vector Space.
        https://arxiv.org/pdf/1301.3781"""
)
parser.add_argument('-b', '--base_dir', nargs='?', type=Path,
                    default=Path(__file__).parent,
                    help='base directory to save/load files to/from.')
parser.add_argument('-m', '--model_file', nargs='?',
                    type=str,  default='cbow.model',
                    help='Where to save/load the model. ')
parser.add_argument('-d', '--data_file', nargs='?',
                    type=str,  default='data/toy_data_41mb_raw.txt.train',
                    help='Where to find the training data.')
parser.add_argument('-o', '--overwrite_model',
                    action=argparse.BooleanOptionalAction,
                    help='Will overwrite existing model or start new model if'
                         'passed. Otherwise looks for model_file in base_dir to '
                         'load and resume training for it.')
parser.add_argument('-t', '--type', nargs='?', type=str,
                    default='cbow', choices=['cbow', 'sgram'],
                    help='Type of model to use. CBOW or S-gram. This will only'
                         ' apply if overwrite_model is used.')
parser.add_argument('-w', '--window', nargs='?', default=7,
                    type=int,
                    help='window size for the model. E.g. for window=5, 2 words'
                         ' on the left and 2 words on the right of the target '
                         'word are used as context.')
parser.add_argument('-v', '--vector_dim', nargs='?', default=300,
                    type=int,
                    help='number of embedding dimensions to use for the cbow')
parser.add_argument('-B', '--batch_size', nargs='?', default=512,
                    type=int)
parser.add_argument('-e', '--epochs', nargs='?', default=1000,
                    type=int)
parser.add_argument('-a', '--alpha', nargs='?', default=0.05,
                    help='learning rate. 0.005 for Cbow and 0.05 for Sgram are'
                         'good to start. Defaults to 0.05',  type=float)

args = parser.parse_args()

DATA_FILE = args.base_dir / args.data_file
MODEL_FILE = args.base_dir / args.model_file


# Wrappers to streamline code in the training loop.
def train_batch(model: Union[Cbow, Sgram], chunk: Tuple[Tuple[List[str], str]]):
    """Wrapper around training steps.

    Chunk: batch size tuple of (List(context), target) pairs.
    """
    c, t = zip(*chunk)
    context = np.array(c).T  # [N-1, B]
    target = np.expand_dims(np.array(t), 0)  # [1, B]
    if type(model) is Cbow:
        model.forward_quick(context)
        model.loss.append(model.loss_fn(target))
    elif type(model) is Sgram:
        model.forward_quick(target)
        model.loss.append(model.loss_fn(context))
    model.backward_quick()
    model.optim_sgd(alpha)


def stats_print(model: Union[Cbow, Sgram], data: Dataloader, batch_counter:int,
                start_time: float, alpha: float):
    """Prints stats to screen and saves them to model obj."""
    print(f'epoch: {model.epoch},   '
          f'line_no {data.line_no:05} of {len(data.data)},   '
          f'time: {(time.time() - start_time) / 60:.6}min,   '
          f'loss:{model.loss[-1]:.5}    '
          )
    # Keep training progress for later analysis of hyperparams.
    model.stats.loc[len(model.stats)] = [
        model.epoch,
        batch_counter,
        batch_counter * model.batch_size,
        (time.time() - start_time) / 60,
        model.loss[-1], alpha]


def stats_save(model: Union[Cbow, Sgram], data: Dataloader,
               args: argparse.Namespace, model_file: str = MODEL_FILE):
    """Saves model and plots to files."""
    model.plot_loss_curve(args.base_dir / f'{args.type}.loss.png')
    model.plot_prediction(data, args.base_dir / f'{args.type}.pred.png')
    with open(model_file, 'wb') as fp:
        pickle.dump({'model': model}, fp)


def load_model(vocab: Vocab, args: argparse.Namespace, seed: int = None
               ) -> Union[Cbow, Sgram]:
    if args.overwrite_model:
        if args.type.lower() == 'cbow':
            model = Cbow(vocab, args.vector_dim, args.window, args.batch_size,
                         seed=seed)
        elif args.type.lower() == 'sgram':
            model = Sgram(vocab, args.vector_dim, args.window, args.batch_size,
                          seed=seed)
        else:
            raise ValueError(f'{args.type=}. Should be in ("cbow", "sgram")')
    else:
        try:
            with open(MODEL_FILE, 'rb') as fp:
                tmp = pickle.load(fp)
            model = tmp['model']
            model._batch_size = args.batch_size
        except FileNotFoundError:
            print(f'{MODEL_FILE} does not exist. Check file name, '
                  'or use overwrite_model arg to start from scratch')
            sys.exit(1)
    return model


if __name__ == '__main__':
    print(f'Working dir: {args.base_dir}')
    print(f'Data File: {DATA_FILE}')
    print(f'Model File: {MODEL_FILE}')
    print(f'Overwrite or start model? {args.overwrite_model}')
    print(f'Learning Rate: {args.alpha}')

    # `train_data` is a list of news stories with variable number of tokens per
    # story.
    with open(DATA_FILE, 'rb') as fp:
        train_data = pickle.load(fp)

    vocab = Vocab(train_data)
    mdl = load_model(vocab, args, seed=1)
    print(f'Model Class: {mdl.__class__}\n')
    start_time = time.time()
    for epoch in range(mdl.epoch, args.epochs):
        batch_counter = 0
        data = Dataloader(train_data, mdl.window)
        batched_data = grouper(data, args.batch_size)

        for chunk in batched_data:
            alpha = args.alpha if epoch < 70 else 0.0008
            train_batch(mdl, chunk)
            batch_counter += 1

            if batch_counter % 10 == 0:
                stats_save(mdl, data, args)
            if batch_counter % 100 == 0 or data.line_no == 1:
                stats_print(mdl, data, batch_counter, start_time, alpha)

        # Save checkpoint at the end of each epoch. Make sure disk has enough
        # space. Might need to delete intermediate checkpoints as you go.
        mdl.epoch += 1
        with open(f'{MODEL_FILE}.{mdl.epoch:04}', 'wb') as fp:
            pickle.dump({'model': mdl}, fp)

    # realistically, the epoch loop isn't completed. Progress is tracked and
    # the program will get manually terminated long before all epochs are run.
    end = time.time()
    with open(MODEL_FILE, 'wb') as fp:
        pickle.dump({'model': mdl}, fp)
