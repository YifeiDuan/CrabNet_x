import torch
from torch.utils.data import Sampler
import random
from torch.utils.data import Dataset
import numpy as np

class CustomWeightedRandomSampler(Sampler):
    def __init__(self, datasets, weights, num_samples):
        self.datasets = datasets
        self.weights = weights
        self.num_samples = num_samples

        dataset_lens = [len(x) for x in self.datasets]
        self.total_length = int(sum([dataset * prob for dataset, prob in zip(dataset_lens, self.weights)]))

        self.sampled_indices = []

        if len(datasets) != len(weights):
            raise ValueError("Number of datasets and weights should be the same")
        
        self.epoch_indices = self._compute_epoch_indices()
        

    def _compute_epoch_indices(self):
        mandatory_inds = []
        non_mandatory_inds = []

        for dataset_idx, dataset in enumerate(self.datasets):
            indices = [(dataset_idx, data_idx) for data_idx in range(len(dataset))]
            if self.weights[dataset_idx] == 1:
                mandatory_inds.extend(indices)
            else:
                non_mandatory_inds.extend(indices)

        remaining_space = self.total_length - len(mandatory_inds)
        indices_ = np.where(np.array(self.weights) != 1)[0]

        for _ in range(remaining_space):
            if len(indices_) == 1:
                dataset_idx = indices_[0]
            else:
                probs = np.array(self.weights)[indices_]
                dataset_idx = np.random.choice(indices_, p=probs/np.sum(probs))
            
            valid_indices = np.array([])
            while len(valid_indices) == 0:
                non_mandatory_array = np.array(non_mandatory_inds)
                condition = (non_mandatory_array[:, 0] == dataset_idx)
                # Get indices that match the condition
                valid_indices = np.where(condition)[0]
                # Select a random index from valid_indices
            random_index = np.random.choice(valid_indices)

            # Get the value from non_mandatory_array using the random index
            data_idx = non_mandatory_array[random_index]
            mandatory_inds.append(tuple(data_idx))
            # Removing the selected index from non_mandatory_inds
            non_mandatory_inds.remove(tuple(data_idx))

        np.random.shuffle(mandatory_inds)

        return mandatory_inds

    def reshuffle_indices(self):
        self.epoch_indices = self._compute_epoch_indices()

    def __iter__(self):
        for idx in range(len(self.epoch_indices)):
            yield self.epoch_indices[idx]

    def __len__(self):
        return self.num_samples

class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights

    def __getitem__(self, index):
        dataset_idx, data_idx = index
        return self.datasets[dataset_idx][data_idx]

    def __len__(self):
        dataset_lens = [len(x) for x in self.datasets]
        return  int(sum([dataset * prob for dataset, prob in zip(dataset_lens, self.weights)]))


