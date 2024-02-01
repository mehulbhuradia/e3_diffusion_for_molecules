import msgpack
import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import random
import json

def train_test_val_split(array, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    random.shuffle(array)
    length = len(array)

    # Calculate the sizes for each split
    train_size = int(length * train_ratio)
    test_size = int(length * test_ratio)

    # Split the array into training, testing, and validation sets
    train_list = array[:train_size]
    test_list = array[train_size:train_size + test_size]
    val_list = array[train_size + test_size:]

    return train_list, test_list, val_list

def split_datasets(path='./processed_big_atoms',max_len=300):
    paths = []
    for pdb in os.listdir(path):
        length=pdb.split('_')[2].split('.')[0]
        if int(length) <= max_len:
            paths.append(path+"/" + pdb)
    train_list, test_list, val_list = train_test_val_split(paths)

    train_dataset = AFDataset(train_list)
    test_dataset = AFDataset(test_list)
    val_dataset = AFDataset(val_list)

    return train_dataset, test_dataset, val_dataset

    
        
    


class AFDataset(Dataset):
  def __init__(self, paths):
    self.paths = paths
    self.lengths = []
    for pdb in paths:
        length=pdb.split('/')[2].split('_')[2].split('.')[0]
        self.lengths.append(int(length))
    # Sort the data list by size
    argsort = np.argsort(self.lengths)               # Sort by decreasing size
    self.paths = [self.paths[i] for i in argsort]
    # Store indices where the size changes
    self.split_indices = np.unique(np.sort(self.lengths), return_index=True)[1][1:]

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    file_path=self.paths[idx]
    with open(file_path, 'r') as file:
      data = json.load(file)
    coords=torch.Tensor(data['coords'])
    one_hot=torch.Tensor(data['one_hot'])
    edges=[]
    for e in data['edges']:
      edges.append(torch.tensor(e, dtype=torch.int64))
    return coords, one_hot, 0, edges, file_path

class CustomBatchSampler(BatchSampler):
    """ Creates batches where all sets have the same size. """
    def __init__(self, sampler, batch_size, drop_last, split_indices):
        super().__init__(sampler, batch_size, drop_last)
        self.split_indices = split_indices

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size or idx + 1 in self.split_indices:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        count = 0
        batch = 0
        for idx in self.sampler:
            batch += 1
            if batch == self.batch_size or idx + 1 in self.split_indices:
                count += 1
                batch = 0
        if batch > 0 and not self.drop_last:
            count += 1
        return count


class AFDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, drop_last=False):

        # This goes over the data sequentially, advantage is that it takes
        # less memory for smaller molecules, but disadvantage is that the
        # model sees very specific orders of data.
        sampler = SequentialSampler(dataset)
        batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last,
                                            dataset.split_indices)
        super().__init__(dataset, batch_sampler=batch_sampler)
