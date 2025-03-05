"""
This script creates a vocabulary for the entire data set (about 4Gb).

At training time only part of the data is loaded, so we have to pre-create the
vocab to ensure our matrices are consistent across the training data.
"""
# Corpus length = 308,535,777 word instances
# Vocab length = 47,002 words
#
# corpus length is 313,878,960 words
# Vocab length is 47,002 words
#
# Process finished with exit code 1

from pathlib import Path
import glob
import pickle
import itertools

import w2v_numpy as w2v

if __name__ == '__main__':

    # read in data files
    BASE_DIR = Path(__file__).parent
    DATA_FOLDER = BASE_DIR / Path('data/processed/thresh_1000')
    data_files = glob.glob(str(DATA_FOLDER) + '/*train[0-9][0-9]')
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
    # Length train_data = 19266872

    # create vocab
    print('Creating vocab')
    vocab = w2v.Vocab(train_data)

    print('saving file')
    with open(DATA_FOLDER / f'vocab.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)
    print('Saved vocab to vocab.pkl')
