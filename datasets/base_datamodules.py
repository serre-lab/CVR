
import os
import math
import random
import argparse

import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader



class DataModuleBase(LightningDataModule):

    def __init__(
        self,  
        num_workers, 
        batch_size, 
    ):
        super().__init__()
        self.batch_size = batch_size

        self.train_transform = None 
        self.test_transform = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.num_workers = num_workers

    def train_dataloader(self):
        # get and process the data first

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # drop_last=True,
            drop_last=False,
        )

    def val_dataloader(self):
        # return both val and test loader

        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return val_loader

    def test_dataloader(self):

        test_dataloader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return test_dataloader

    @property
    def n_classes(self):
        # self._n_class should be defined in _prepare_train_dataset()
        return self._n_classes

    @property
    def num_train_data(self):
        assert self.train_set is not None, (
            "Load train data before calling %s" % self.num_train_data.__name__
        )
        return len(self.train_set)

    @property
    def num_val_data(self):
        assert self.train_set is not None, (
            "Load train data before calling %s" % self.num_val_data.__name__
        )
        return len(self.val_set)

    @property
    def num_test_data(self):
        assert self.test_set is not None, (
            "Load test data before calling %s" % self.num_test_data.__name__
        )
        return len(self.test_set)

