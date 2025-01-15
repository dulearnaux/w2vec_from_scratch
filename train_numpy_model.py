from typing import Union

from w2v_numpy import *

import pickle
from pathlib import Path
import argparse
import sys
import time
import glob
import itertools

# Input args:
#  -b= path to base directory
#  -m= model filename. Relative to base directory.
#  -d= Training data filenames. Relative to base directory. A glob can be passed
#      to ID multiple filenames.
#  -o, include to overwrite existing model or start new model. Exclude to load
#       existing.
#  -t= ['cbow', 'sgram']. Use CBOW or SGRAM model. Defaults to CBOW.
#  -w= window length to use. must be odd and >= 3.
#  -k= number of negative samples to use. Its ignored if model is not SgramNS
#      or CbowNS
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
                    type=str, default='cbow.model',
                    help='Where to save/load the model. ')
parser.add_argument('-d', '--data_file', nargs='?',
                    type=str, default='data/*.train[0-9][0-9]',
                    help='Where to find the training data. Can pass a glob.')
parser.add_argument('-o', '--overwrite_model',
                    action=argparse.BooleanOptionalAction,
                    help='Will overwrite existing model or start new model if'
                         'passed. Otherwise looks for model_file in base_dir to '
                         'load and resume training for it.')
parser.add_argument('-t', '--type', nargs='?', type=str,
                    default='cbow', choices=['cbow', 'sgram', 'cbow_neg', 'sgram_neg'],
                    help='Type of model to use. CBOW or S-gram. This will only'
                         ' apply if overwrite_model is used.')
parser.add_argument('-w', '--window', nargs='?', default=7,
                    type=int,
                    help='window size for the model. E.g. for window=5, 2 words'
                         ' on the left and 2 words on the right of the target '
                         'word are used as context.')
parser.add_argument('-k', '--neg_samples', nargs='?', default=5,
                    type=int,
                    help='Number of negative samples to use. Only applies if '
                         'type is in [cbow_neg, sgram_neg]')
parser.add_argument('-v', '--vector_dim', nargs='?', default=300,
                    type=int,
                    help='number of embedding dimensions to use for the cbow')
parser.add_argument('-B', '--batch_size', nargs='?', default=512,
                    type=int)
parser.add_argument('-e', '--epochs', nargs='?', default=1000,
                    type=int)
parser.add_argument('-a', '--alpha', nargs='?', default=0.05,
                    help='learning rate. 0.005 for Cbow and 0.05 for Sgram are'
                         'good to start. Defaults to 0.05', type=float)

args = parser.parse_args()

DATA_FILE = Path(args.base_dir) / Path(args.data_file)
MODEL_FILE = Path(args.base_dir) / Path(args.model_file)


# Wrappers to streamline code in the training loop.
def train_batch(model: Union[Cbow, Sgram, CbowNS, SgramNS],
                chunk: Tuple[Tuple[List[str], str]], alpha: float):
    """Wrapper around training steps.

    Chunk: batch size tuple of (List(context), target) pairs.
    """
    c, t = zip(*chunk)
    context = np.array(c).T  # [N-1, B]
    target = np.expand_dims(np.array(t), 0)  # [1, B]
    if type(model) is Cbow:
        model.forward_quick(context)
        model.loss.append(model.loss_fn(target))
        model.backward_quickest()
    elif type(model) is Sgram:
        model.forward_quick(target)
        model.loss.append(model.loss_fn(context))
        model.backward_quickest()
    elif type(model) is CbowNS:
        neg_words = data.neg_samples(context, target)
        model.forward_neg_quick(target, context, neg_words)
        model.loss.append(model.loss_fn_neg())
        model.backward_neg_quick()
    elif type(model) is SgramNS:
        neg_words = data.neg_samples(context, target)
        model.forward_neg_quick(target, context, neg_words)
        model.loss.append(model.loss_fn_neg())
        model.backward_neg_quickest()
    model.optim_sgd(alpha)


def stats_print(model: Union[Cbow, Sgram, CbowNS], data: Dataloader, batch_counter: int,
                start_time: float, alpha: float, data_num: int):
    """Prints stats to screen and saves them to model obj."""
    if type(model) in (CbowNS, SgramNS):
        # This loss is comparable to Cbow and Sgram
        loss_normalized = model.loss_normalized()
    else:
        loss_normalized = model.loss[-1]
    print(f'epoch: {model.epoch},   '
          f'data: {data_num:02}'
          f'line_no {data.line_no:07} of {len(data.data)},   '
          f'time: {(time.time() - start_time) / 60:.6}min,   '
          f'loss:{model.loss[-1]:.5}    '
          f'loss (normalize):{loss_normalized:.5}    '
          )
    # Keep training progress for later analysis of hyperparams.
    model.stats.loc[len(model.stats)] = [
        model.epoch,
        batch_counter,
        batch_counter * model.batch_size,
        (time.time() - start_time) / 60,
        model.loss[-1],
        loss_normalized,
        alpha]


def stats_save(model: Union[Cbow, Sgram, CbowNS, SgramNS], data: Dataloader,
               args: argparse.Namespace, model_file: str = MODEL_FILE):
    """Saves model and plots to files."""
    model.plot_loss_curve(args.base_dir / f'{args.type}.loss.png')
    model.plot_prediction(data, args.base_dir / f'{args.type}.pred.png')
    with open(model_file, 'wb') as fp:
        pickle.dump({'model': model}, fp)


def load_model(vocab: Vocab, args: argparse.Namespace, seed: int = None
               ) -> Union[Cbow, Sgram, CbowNS, SgramNS]:
    if args.overwrite_model:
        if args.type.lower() == 'cbow':
            model = Cbow(vocab, args.vector_dim, args.window, args.batch_size,
                         seed=seed)
        elif args.type.lower() == 'sgram':
            model = Sgram(vocab, args.vector_dim, args.window, args.batch_size,
                          seed=seed)
        elif args.type.lower() == 'cbow_neg':
            model = CbowNS(vocab, args.vector_dim, args.window, args.batch_size,
                           seed=seed)
        elif args.type.lower() == 'sgram_neg':
            model = SgramNS(vocab, args.vector_dim, args.window, args.batch_size,
                            seed=seed)
        else:
            raise ValueError(f'{args.type=}. Should be in '
                             f'("cbow", "sgram", "cbow_neg", "sgram_neg)')
    else:
        try:
            with open(MODEL_FILE, 'rb') as fp:
                tmp = pickle.load(fp)
            model = tmp['model']
            model._batch_size = args.batch_size
            print(f'{MODEL_FILE} Successfully loaded')
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

    data_files = glob.glob(str(DATA_FILE))
    data_files.sort()
    print('Data files identified')
    for data_file in data_files:
        print(data_file)


    # For sgram, need negative samples for each context (output) word.
    neg_sample_adj = ((args.window - 1) if args.type == 'sgram_ns' else 1)

    # load vocab
    with open('vocab.pkl', 'rb') as fp:
        vocab = pickle.load(fp)

    # load model
    mdl = load_model(vocab, args, seed=1)
    print(f'Model Class: {mdl.__class__}\n')

    epoch_start = mdl.epoch
    start_time = time.time()
    batch_counter = 0
    for epoch in range(epoch_start, args.epochs):
        for data_num, file in enumerate(data_files):
            # Slow down printing for later iters, and for neg sampling
            iteration_adj = (10 if type(args.type) in ('cbow_ns', 'sgram_ns') else 1)
            iteration_adj *= (1 if (data_num + epoch) == 0 else 100)

            with open(file, 'rb') as fp:
                train_data = pickle.load(fp)
            # Flatten list to contain one long list of lines.
            train_data = list(itertools.chain(*train_data))

            # neg samples will be ignored unless data.neg_samples() is directly called.
            data = Dataloader(
                train_data, mdl.window,
                negative_samples=args.neg_samples * neg_sample_adj)
            batched_data = grouper(data, args.batch_size)

            for chunk in batched_data:
                alpha = args.alpha if epoch < 5 else 0.0008
                train_batch(mdl, chunk, alpha)
                batch_counter += 1

                if batch_counter % (10 * iteration_adj) == 0:
                    stats_save(mdl, data, args)
                if batch_counter % (100 * iteration_adj) == 0 or data.line_no == 1:
                    stats_print(
                        mdl, data, batch_counter, start_time, alpha, data_num)

            # Save checkpoint at the end of each data file.
            with open(f'{MODEL_FILE}.data.{data_num:02}', 'wb') as fp:
                pickle.dump({'model': mdl}, fp)
            # Keep only the last 1 checkpoints.
            Path(f'{MODEL_FILE}.data.{data_num - 1:02}').unlink(missing_ok=True)
            # incase it's the first data file of the epoch.
            Path(f'{MODEL_FILE}.data.{data_num + 9:02}').unlink(missing_ok=True)

        # Save checkpoint at the end of each epoch. Make sure disk has enough
        # space. Might need to delete intermediate checkpoints as you go.
        mdl.epoch += 1
        with open(f'{MODEL_FILE}.epoch.{mdl.epoch:04}', 'wb') as fp:
            pickle.dump({'model': mdl}, fp)
        # Keep only the last 3 checkpoints.
        Path(f'{MODEL_FILE}.epoch.{mdl.epoch - 3:04}').unlink(missing_ok=True)

    # realistically, the epoch loop isn't completed. Progress is tracked and
    # the program will get manually terminated long before all epochs are run.
    end = time.time()
    with open(MODEL_FILE, 'wb') as fp:
        pickle.dump({'model': mdl}, fp)
