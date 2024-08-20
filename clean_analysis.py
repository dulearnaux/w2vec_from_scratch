"""
This script preprocess a sample of the data (1 of 100 files). Some analysis is
run to inform how the overall data set will ultimately get processes.
"""

from typing import List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from unidecode import unidecode
import string

from pathlib import Path
import pickle

LOAD_CLEAN = True  # if true, will load saved clean data and not process new.
LOAD_TRAIN = False  # if true, will load saved training data and not process new.
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / 'data/toy_data_41mb_raw.txt'
LOW_FREQ_THRESHOLD = 10  # minimum frequency threshold for words to include in vocab
STEM_WORDS = False  # To use only word stems set to True
# Some sequences (individual news stories) are very short. This parameter
# removes news stories with low number of tokens. Note, the shortest sequences
# will ultimately become even shorter after stopwords are removed and other
# processing completed.
MIN_SEQUENCE_LENGTH = 10


def clean_line(line: str) -> List[str]:
    """Cleans a single line of text."""

    # apostrophe words have a space before the apostrophe for some reason.
    line = line.replace(" '", "'")

    # remove punctuation
    puncs = string.punctuation
    table = str.maketrans('', '', puncs)
    stripped = line.translate(table)

    # map non-english characters to english ones. e.g. â to a.
    stripped = unidecode(stripped)

    # create tokens
    tokens = word_tokenize(stripped)
    tokens = [word.lower() for word in tokens]
    words = [word for word in tokens if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    if STEM_WORDS:
        words = [clean_stem(word) for word in words]

    return words


def clean_stem(tokens: Sequence[str]) -> List[str]:
    """Stemming isn't necessary for word embeddings, but it can reduce our
    vocab size. So we can do this optionally."""
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    return stemmed


def token_freq(lines: Sequence[Sequence[str]])\
        -> Tuple[npt.NDArray[str], npt.NDArray[int]]:
    """Returns token frequency"""
    sentences = [' '.join(word) for word in lines]
    sentences = ' '.join(sentences)
    tokens = np.array(sentences.split(' '))
    token, freq = np.unique(tokens, return_counts=True)
    order_index = np.argsort(freq)[::-1]  # high to low freq
    return token[order_index], freq[order_index]


def remove_oov_words(
        clean_lines: Sequence[Sequence[str]]) -> Sequence[Sequence[str]]:
    """Removes the words that are out of vocabulary.

    Resorts to numpy arrays to reduce processing time. Might use up a lot
    more memory but it 100x faster than list iterations."""
    flat_text = ' \n '.join([' '.join(line) for line in clean_lines])
    tokens = np.array(flat_text.split(' '))
    tmp_vocab = np.append(vocab, '\n')  # preserve the newlines
    tokens = tokens[np.isin(tokens, tmp_vocab)]
    lines = ' '.join(tokens.tolist()).split(' \n ')
    return [line.split(' ') for line in lines]


if __name__ == '__main__':

        clean_data_file = DATA_FILE.parent / (DATA_FILE.name + '.clean')
        # Load in already cleaned data
        if LOAD_CLEAN:
            with open(clean_data_file, 'rb') as fp:
                clean_lines = pickle.load(fp)
        # else, clean and save data
        else:
            with open(DATA_FILE, 'rt') as file:
                text = file.read()
            # Lines appear to be split based on news story. Each line is an
            # independent news story.
            lines = text.split('\n')
            # generate a list of a list of tokens.
            clean_lines = [clean_line(line) for line in lines]
            with open(clean_data_file, 'wb') as fp:
                pickle.dump(clean_lines, fp)

        # Analyse the vocab
        print(f'ANALYSING THE DATA\n')
        # Some lines have very few tokens in them. E.g. 1 token in a line. Remove
        # lines that have fewer than MIN_SEQUENCE_LENGTH words
        line_lengths = np.array([len(line) for line in clean_lines])
        num_short_sequences = sum(line_lengths < MIN_SEQUENCE_LENGTH)
        corpus = ' '.join([' '.join(words) for words in clean_lines])
        corpus_tokens = corpus.split(' ')
        corpus_len_orig = len(corpus_tokens)
        clean_lines = [line for line in clean_lines if (len(line) >= MIN_SEQUENCE_LENGTH)]

        corpus = ' '.join([' '.join(words) for words in clean_lines])
        corpus_tokens = corpus.split(' ')
        corpus_len = len(corpus_tokens)
        print(f'Number of short news stories with fewer than {MIN_SEQUENCE_LENGTH}'
              f' words: {num_short_sequences:,} of {len(clean_lines):,}')
        print(f'Corpus size reduced from {corpus_len_orig:,} to {corpus_len:,} after '
              f'removing short sequences.')

        # low to high freq sorted pd series.
        tokens, freqs = token_freq(clean_lines)
        assert len(tokens) == len(set(tokens)), (
            f'output from token_freq is not a set')
        vocab_size = len(tokens)
        num_low_freq_words = sum(freqs < LOW_FREQ_THRESHOLD)
        high_freq_vocab = sum(freqs >= LOW_FREQ_THRESHOLD)
        high_freq_occurrences = freqs[freqs >= LOW_FREQ_THRESHOLD].sum()
        high_freq_occurrences_pct = high_freq_occurrences / freqs.sum()
        stem_only_vocab_len = len(set(clean_stem(tokens[freqs>=LOW_FREQ_THRESHOLD])))

        print(f'Number of words in corpus: {corpus_len:,}')
        print(f'Corpus vocab size: {vocab_size:,}')
        print(f'Number of words with freq < {LOW_FREQ_THRESHOLD}: {num_low_freq_words:,}')
        print(f'Number of words with freq >= {LOW_FREQ_THRESHOLD}: {high_freq_vocab:,}')
        print(f'Number of word instances with freq > '
              f'{LOW_FREQ_THRESHOLD}: {high_freq_occurrences_pct:.2%}')
        print(f'Vocab size using word stems (high freq only): {stem_only_vocab_len:,}')

        # Plot cumulative word instances vs vocab size
        import matplotlib
        # matplotlib.use('Agg')
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt

        cuml = freqs.cumsum() / freqs.sum()
        fig = plt.figure()
        plt.plot(np.arange(len(cuml)), cuml)
        plt.title('Cumulative Percentage of Word Instances vs Word Frequency')
        plt.ylabel('Percentage of all word instances (as pct of corpus length)')
        plt.xlabel('Vocab Size')
        plt.show()

        # We can trim the vocab from 149k to 25k words and keep 90% of the word
        # impressions!
        vocab = tokens[freqs >= LOW_FREQ_THRESHOLD]

        # remove non-vocab words from clean data and save to file.
        train_data_file = DATA_FILE.parent / (DATA_FILE.name + '.train')
        if LOAD_TRAIN:
            with open(train_data_file, 'rb') as fp:
                train_data = pickle.load(fp)
        else:
            train_data = remove_oov_words(clean_lines)
            with open(train_data_file, 'wb') as fp:
                pickle.dump(train_data, fp)

        corpus_len_trimmed = len(' '.join([' '.join(x) for x in train_data]).split(' '))
        print(f'Corpus length after trimming low freq words: {corpus_len_trimmed:,}'
              f'\nReduced by : {1 - corpus_len_trimmed/corpus_len:0.2%}')
