"""
This script creates a vocabulary for the entire data set (about 4Gb).

At training time only part of the data is loaded, so we have to pre-create the
vocab to ensure our matrices are consistent across the training data.
"""

from pathlib import Path
import glob
import pickle
import itertools

import w2v_numpy as w2v

if __name__ == '__main__':

    # read in data files
    BASE_DIR = Path(__file__).parent
    data_files = glob.glob(str(BASE_DIR / Path('data/*train')) + ('[0-9]' * 2))
    data_files.sort()
    print('Data files identified')
    for data_file in data_files:
        print(data_file)

    data = []
    for data_file in data_files:
        with open(data_file, 'rb') as f:
            data.append(pickle.load(f))

    # Flatten list to contain one long list of lines.
    train_data = list(itertools.chain(*itertools.chain(*data)))
    print(f'Length train_data = {len(train_data)}')

    # create vocab
    print('Creating vocab')
    vocab = w2v.Vocab(train_data)

    print('saving file')
    with open(BASE_DIR / 'vocab.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)
    print('Saved vocab to vocab.pkl')
