from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data
import pickle
import warnings
import cv2
import numpy as np
warnings.filterwarnings("ignore")



def get_distr_dataloader(paths_file, root_dir, rank, world_size, transform=None, batch_size=32, num_workers=4,
                         shuffle=True):
    """
    return a distributed dataloader
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                    my_collate=None, pseudo_path=False, is_dist=False):
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, pseudo_path=pseudo_path)
    
    if is_dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=True, sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)

    return dataloader


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None, multilabel=False, pseudo_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(csv_file) == str:
            self.data_paths = pd.read_csv(csv_file, header=None)
        elif type(csv_file) in [list, tuple]:
            self.data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.pseudo_path = pseudo_path

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0])

        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        if self.pseudo_path:
            pic_name = os.path.basename(img_name)
            pic_name, _ = os.path.splitext(pic_name)
            label_path = os.path.join(self.pseudo_path, pic_name+".png")
            label = np.array(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
            sample["labels"] = label

        if self.transform:
            sample = self.transform(sample)

        sample["img_path"] = img_name       
        return sample

    def read(self, idx, abs=False):
        """
        read single dataset sample corresponding to idx (index number) without any data transform applied
        """
        if type(idx) == int:
            img_name = os.path.join(self.root_dir,
                                    self.data_paths.iloc[idx, 0])
        if type(idx) == str:
            if abs:
                img_name = idx
            else:
                img_name = os.path.join(self.root_dir, idx)
        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        return sample
    
    
def my_collate(batch):
    "Filter out sample where mask is zero everywhere"
    idx = [b['unk_masks'].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)
