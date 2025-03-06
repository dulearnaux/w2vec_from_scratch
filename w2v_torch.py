from typing import Sequence, List

from pathlib import Path
import pickle
import itertools
import time
import os

import numpy as np
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import torch

import w2v_numpy as w2v

# Create 3 separate IterableDatasets
#  * intra file iterator:  to iterate over a single file
#  * inter file iterator: to iterate over files
#  * combined: iterates over all items within a file, and then over all files.
#       This is our final iterator to be passed into Dataloader
class IntraFileDataset(IterableDataset):
    """Iterates through single data file."""
    def __init__(self, file: str, vocab, window: int, device=torch.device('cpu')):
        super().__init__()

        with open(file, 'rb') as fp:
            print(f'creating iterator from file: {file}')
            data = list(itertools.chain(*pickle.load(fp)))

        self.iterator = w2v.Dataloader(data, window)
        self.device = device
        self.vocab = vocab
        self.file = file

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        return self

    def __next__(self):
        c, t = next(self.iterator)
        context = np.array(c)  # [B, N-1]
        target = np.array(t)  # [B, 1]
        # Embedding layer takes index as input, not OHE.
        context = self.vocab.encode_idx(context[:, np.newaxis]) 
        target = self.vocab.encode_idx_fast(target.reshape((1, 1)))
        return torch.as_tensor(context).to(self.device), torch.as_tensor(target).to(self.device)  # single batch


class InterFileDataset(IterableDataset):
    """Iterates over files.

    Iterates across each file to create a IntraFileDataset for each file, but
    doesn't iterate through the IntraFileDataset.
    """

    def __init__(self, files: List[str], vocab, window: int, device=torch.device('cpu')):
        super().__init__()

        self.files = files.copy()
        self.file = files.pop(0)
        self.files_remaining = files

        self.intra_file_iterator = IntraFileDataset(self.file, vocab, window, device)
        self.device = device
        self.window = window
        self.vocab = vocab

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Get next intra file iterator if it's available.
            self.file = self.files_remaining.pop(0)
            self.intra_file_iterator = IntraFileDataset(
                self.file, self.vocab, self.window, self.device)
        except IndexError:
            raise StopIteration

        return self.intra_file_iterator


class CombinedDataset(IterableDataset):
    """Combines IntraFileDataset and InterFileDataset.

    Iterates though each file, and across all files.
    """
    def __init__(self, files: List[str], vocab, window: int, device=torch.device('cpu')):
        super().__init__()
        self.inter_file_iterator = InterFileDataset(files, vocab, window, device)
        self.device = device

    def __len__(self):
        # Approximate length. Would need to read every file to get actual length.
        return len(self.inter_file_iterator)*len(self.inter_file_iterator.intra_file_iterator)

    def __iter__(self):
        return self

    def __next__(self):

        try:
            context, target = next(self.inter_file_iterator.intra_file_iterator)
        except StopIteration:
            self.inter_file_iterator.intra_file_iterator = next(self.inter_file_iterator)
            context, target = next(self.inter_file_iterator.intra_file_iterator)
        return context.to(self.device), target.to(self.device)


def worker_init_fn(worker):
    """Initializes iterator dataset for each worker.

    Divides the files from the main, into the workers copy
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    num_workers = worker_info.num_workers
    dataset = worker_info.dataset  # Workers copy of dataset
    all_files = dataset.inter_file_iterator.files  # All the files.

    worker_files = []
    for i, file in enumerate(all_files):
        if i % num_workers == worker_id:
            worker_files.append(file)
    # Re-init inter_file_iterator with new list of worker_files

    print(f'Reinitializing InterFileDataset for {worker_id=}')
    print(f'Worker_id={worker_id}, files={", ".join([os.path.basename(file) for file in worker_files])}')
    dataset.inter_file_iterator = InterFileDataset(
        worker_files,
        dataset.inter_file_iterator.vocab,
        dataset.inter_file_iterator.window,
        dataset.inter_file_iterator.device)


class CbowTorch(nn.Module):
    """Pytorch embedding model for word23vec.

    Uses embedding layer to extract an embedding for each input word. Takes the
    average for each input word, so we have one embedding per batch. Uses
    linear layer to map to output word.
    """
    def __init__(
            self, vocab: w2v.Vocab, embed_dim: int, alpha, 
            device=torch.device('cpu')):
        super().__init__()
        self.embedding = nn.Embedding(vocab.size, embed_dim).to(device)
        self.linear = nn.Linear(embed_dim, vocab.size, bias=False).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.losses = []
        self.device = device

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # X has multiple words per input, we take the average embedding for all words.
        x = self.linear(x)
        return x

    def train_step(self, x: torch.tensor, y: torch.tensor, device=torch.device('cpu')):

        y_pred = self.forward(x.to(device))
        loss = self.loss_fn(y_pred, y.to(device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Save for later
        self.losses.append(loss.item())


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    torch.set_default_device(device)
    torch.manual_seed(0)

    base_dir = Path(__file__).parent
    data_files = list((base_dir / Path('data/processed/thresh_1000/')).glob('news.train[0-9][0-9]'))
    # data_files = list((base_dir / Path('data/processed/test/')).glob('news.train[0-9][0-9]'))
    vocab_file = base_dir / Path('data/processed/thresh_1000/vocab.pkl')
    data_files.sort()
    window = 7
    batch_size = 1024*4
    embed_dim = 300
    alpha = 0.0001

    with open(vocab_file, 'rb') as fp:
        vocab = pickle.load(fp)
    dataset = CombinedDataset(data_files, vocab, window, device)
    model = CbowTorch(vocab, embed_dim, alpha, device)

    # https://github.com/pytorch/pytorch/issues/40403
    # Issue with pytorch, CUDA and multiprocessing. Must use spawn method if
    # using multiple workers.
    num_workers = 10
    if num_workers > 0:
        torch.multiprocessing.set_start_method('spawn')
        prefetch_factor = 10
        fn = worker_init_fn
    else:
        prefetch_factor = None
        fn = None
    batched_data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=False,
        worker_init_fn=fn,
    )
    start = time.perf_counter()
    batch_start_time = start
    num_batches = len(batched_data)
    for batch_counter, (context, target) in enumerate(batched_data):
        context = context.squeeze().to(device)
        target = target.squeeze().to(device)
        model.train_step(x=context, y=target, device=device)
        print_increment = 100
        if batch_counter % print_increment == 0 and batch_counter > 0:
            dur = (time.perf_counter() - start)/60
            batch_dur = (time.perf_counter() - batch_start_time)
            batch_start_time = time.perf_counter()  # reset batch timer
            print(f'batch = {batch_counter} of {num_batches},  '
                  f'loss = {model.losses[-1]:.5},   \
                  time:{dur:.5},   '
                  f'epoch_dur_est = {dur/(batch_counter/num_batches):.5},   '
                  f'items/s = {(batch_size * print_increment) /batch_dur:.5}')

    print('Pytorch Word2vec complete')

