"""
This script cleans the entire corpus of data. About 4Gb.
 - removes punctuation and odd letters
 - tokenises
 - removes stop words
 - down samples common words according to Mikolov et al.
 - removes OOV words.
 - saves the processed data.
"""
from typing import List, Sequence, Tuple

from pathlib import Path
import pickle
import os
from random import shuffle

from unidecode import unidecode
import string
from collections import Counter
from multiprocessing import Pool

import numpy as np
import numpy.typing as npt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


BASE_DIR = Path(__file__).parent
LOW_FREQ_THRESHOLD = 1000  # minimum frequency threshold for words to include in vocab
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

    # load data, about 4Gb, can do it in-memory
    def get_data_from_file(file):
        toc = time.perf_counter()
        print(f'cleaning {file}, time={start/60-toc/60:.2f} minutes, ')
        with open(file, 'rt') as fl:
            text = fl.read()
        lines = text.split('\n')
        # Convert to lines of tokens
        clean_lines = [clean_line(line) for line in lines]
        # remove short lines (sentences to short to train on).
        clean_lines = [line for line in clean_lines if
                       (len(line) >= MIN_SEQUENCE_LENGTH)]
        return clean_lines

    import time
    start = time.time()
    N_CORES = os.cpu_count()
    with Pool(N_CORES-1) as pool:
        data = pool.map(get_data_from_file, data_files)

    # get vocab. vocab_raw holds {words: freq} pairs.
    vocab_raw = Counter()
    for clean_lines in data:
        sentences = [' '.join(word) for word in clean_lines]
        sentences = ' '.join(sentences)
        freqs = Counter(sentences.split(' '))
        vocab_raw += freqs

    # Print out a summary of vocab size vs LOW_FREQ_THRESHOLD
    vocab_summary_file = BASE_DIR / Path("data/processed/vocab_summary.txt")
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    os.remove(vocab_summary_file) if os.path.exists(vocab_summary_file) else None
    for threshold in [0, 10, 100, 1000, 10_000, 100_000]:
        vocab = Counter({k: c for k, c in vocab_raw.items() if c >= threshold})
        output = f'Freq Threshold = {threshold:_}, Vocab length = {len(vocab):_}, Corpus length = {sum(vocab.values()):_}'
        print(output)
        with open(vocab_summary_file, 'a') as summary:
            summary.write(output+'\n')

    # Set the vocab for out LOW_FREQ_THRESHOLD
    vocab = Counter({k: c for k, c in vocab_raw.items() if c >= LOW_FREQ_THRESHOLD})

    total = sum(vocab.values())
    print(f'Corpus length = {total:,} word instances')
    print(f'Vocab length = {len(vocab):,} words')
    vocab_probs = {k: v / total for k, v in vocab.items()}

    # create discard probabilities according to Mikolov et. al. Section 2.3
    vocab_adj_probs = {word: (1 - np.sqrt(1e5/probs)) for word, probs in vocab_probs.items()}
    total = sum(vocab_adj_probs.values())
    vocab_adj_probs = Counter({k: v / total for k, v in vocab_adj_probs.items()})

    pre_corpus = 0
    post_corpus = 0
    # Subsample words according to vocab_adj_probs
    # Remove OOV words.
    for i, clean_lines in enumerate(data):
        for j, line in enumerate(clean_lines):
            unifs = np.random.uniform(0, 1, size=len(line))
            pre_corpus += len(line)
            clean_lines[j] = [word for word, u in zip(line, unifs) if vocab_adj_probs[word] < u and word in vocab.keys()]
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

    def split_list(lst, n, to_shuffle=True):
        if to_shuffle:
            # Does a full corpus shuffle of news stories. E.g. news story at end
            # of last file can be in the beginning of the first file now.
            shuffle(lst)
        return [lst[i::n] for i in range(n)]

    # Save processed data.
    # Saves in data/processed/thresh_{threshold} folder
    save_data = split_list(data, min(100, len(data)))
    data_save_path = BASE_DIR / Path(f'data/processed/thresh_{LOW_FREQ_THRESHOLD}')
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    for i, dat in enumerate(save_data):
        with open(data_save_path / Path(f'news.train{i:02}'), 'wb') as fp:
            pickle.dump(dat, fp)
