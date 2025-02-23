"""
This script cleans the entire corpus of data. About 4Gb.
 - removes punctuation and odd letters
 - tokenises
 - removes stop words
 - down samples common words according to Mikolov et al.
 - saves the processed data.
"""
from typing import List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from unidecode import unidecode
import string
from collections import Counter

from pathlib import Path
import pickle
import os

LOAD_CLEAN = False  # if true, will load saved clean data and not process new.
LOAD_TRAIN = False  # if true, will load saved training data and not process new.
BASE_DIR = Path(__file__).parent
LOW_FREQ_THRESHOLD = 1000  # minimum frequency threshold for words to include in vocab
STEM_WORDS = True  # To use only word stems set to True



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
        words = clean_stem(words)

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

def token_freq2(lines):
    sentences = [' '.join(word) for word in lines]
    sentences = ' '.join(sentences)
    c = Counter(sentences.split(' '))
    token, freq = np.array(list(c.keys())), np.array(list(c.values()))
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

    data_files = []
    for file in os.listdir(BASE_DIR / Path("data/raw")):
        data_files.append(BASE_DIR / Path("data/raw") / Path(file))
    data_files.sort()


    data = []
    # load data, about 4Gb, can do it in-memory
    for file in data_files:
        print(f'cleaning {file}')
        with open(file, 'rt') as fl:
            text = fl.read()
        lines = text.split('\n')
        # Convert to lines of tokens
        clean_lines = [clean_line(line) for line in lines]
        # remove short lines (sentences to short to train on).
        clean_lines = [line for line in clean_lines if (len(line) >= MIN_SEQUENCE_LENGTH)]
        data.append(clean_lines)

    # get vocab
    vocab_raw = Counter()
    for clean_lines in data:
        sentences = [' '.join(word) for word in clean_lines]
        sentences = ' '.join(sentences)
        freqs = Counter(sentences.split(' '))
        vocab_raw += freqs

    vocab = Counter({k: c for k, c in vocab_raw.items() if c >= LOW_FREQ_THRESHOLD})

    total = sum(vocab.values())
    print(f'Corpus length = {total:,} word instances')
    print(f'Vocab length = {len(vocab):,} words')
    # Corpus length = 295,895,420 word instances
    # Vocab length = 15,358 words
    vocab_probs = {k: v / total for k, v in vocab.items()}

    # create discard probabilities according to Mikolov et. al. Section 2.3
    vocab_adj_probs = {word: (1 - np.sqrt(1e5/probs)) for word, probs in vocab_probs.items()}
    total = sum(vocab_adj_probs.values())
    vocab_adj_probs = Counter({k: v / total for k, v in vocab_adj_probs.items()})

    pre_corpus = 0
    post_corpus = 0
    for i, clean_lines in enumerate(data):
        for j, line in enumerate(clean_lines):
            unifs = np.random.uniform(0, 1, size=len(line))
            pre_corpus += len(line)
            clean_lines[j] = [word for word, u in zip(line, unifs) if vocab_adj_probs[word] < u]
            post_corpus += len(clean_lines[j])

    print(f'pre_corpus length is {pre_corpus:,} words')
    print(f'post_corpus length is {post_corpus:,} words')
    # pre_corpus length is 313,759,360 words
    # post_corpus length is 313,753,782 words

    # length data:
    corpus = 0
    for cl in data:
        for line in cl:
            corpus += len(line)
    print(f'corpus length is {corpus:,} words')
    print(f'Vocab length is {len(vocab):,} words')
    # corpus length is 313,753,782 words
    # Vocab length is 15,358 words

    def split_list(lst, n):
        return [lst[i::n] for i in range(n)]

    save_data = split_list(data, 10)
    for i, dat in enumerate(save_data):
        with open(BASE_DIR / Path(f'data/news.train{i:02}'), 'wb') as fp:
            pickle.dump(dat, fp)

