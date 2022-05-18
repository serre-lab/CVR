import os
import argparse

import torch

from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms as tvt

from datasets.base_datamodules import DataModuleBase
from datasets import transforms_ssl

# from datasets import transforms as all_transforms

from PIL import Image


class CVRTSSLDataModule(DataModuleBase):

    def __init__(
        self,
        data_dir,
        task,
        train_transform,
        test_transform,
        n_samples,
        num_workers,
        batch_size,
        # image_size,
        **kwargs,
    ):

        super(CVRTSSLDataModule, self).__init__(   num_workers,
                                                batch_size,
        )

        # if train_transform in vars(all_transforms):
        #     self.train_transform = vars(all_transforms)[train_transform]()
        # else:
        #     self.train_transform = self._default_transforms()
        
            
        # if test_transform in vars(all_transforms):
        #     self.test_transform = vars(all_transforms)[train_transform]()
        # else:
        #     self.test_transform = self._default_transforms()
        
        # self.train_transform = self._default_transforms()        
        # self.test_transform = self._default_transforms()
        
        transform = self._default_transforms()

        self.train_set = CVRT(data_dir, task, split='train', n_samples=n_samples, image_size=128, transform=transform)
        self.val_set = CVRT(data_dir, task, split='val', n_samples=-1, image_size=128, transform=transform)
        self.test_set = CVRT(data_dir, task, split='test', n_samples=-1, image_size=128, transform=transform)

        # self.setup()

    def get_train_labels(self):
        return self.train_set.targets

    def _default_transforms(self):
        
        # normalize = tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = tvt.Normalize(mean=(0.9, 0.9, 0.9), std=(0.1, 0.1, 0.1))

        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        augmentation1 = [
            tvt.RandomResizedCrop(128, scale=(0.8, 1.)),
            tvt.RandomApply([
                tvt.ColorJitter(0, 0, 0, 0.5)  # not strengthened
            ], p=0.8),
            tvt.RandomGrayscale(p=0.2),
            tvt.RandomApply([transforms_ssl.GaussianBlur([.1, 2.])], p=1.0),
            tvt.RandomHorizontalFlip(),
            tvt.RandomVerticalFlip(),
            transforms_ssl.RotationTransform([-90, 0, 90, 180]),
            tvt.ToTensor(),
            normalize
        ]

        augmentation2 = [
            tvt.RandomResizedCrop(128, scale=(0.8, 1.)),
            tvt.RandomApply([
                tvt.ColorJitter(0, 0, 0, 0.5)  # not strengthened
            ], p=0.8),
            tvt.RandomGrayscale(p=0.2),
            tvt.RandomApply([transforms_ssl.GaussianBlur([.1, 2.])], p=1.0),
            tvt.RandomHorizontalFlip(),
            tvt.RandomVerticalFlip(),
            transforms_ssl.RotationTransform([-90, 0, 90, 180]),
            tvt.ToTensor(),
            normalize
        ]
        
        return transforms_ssl.TwoCropsTransform(tvt.Compose(augmentation1), 
                                                tvt.Compose(augmentation2))



    @staticmethod
    def add_dataset_specific_args(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--data_dir', type=str, default='')
        parser.add_argument('--train_transform', type=str, default='')
        parser.add_argument('--test_transform', type=str, default='')
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--task', type=str, default='0')
        parser.add_argument('--n_samples', type=int, default=-1)

        return parser





TASKS={
    ### elementary
    0: "task_shape",
    1: "task_pos",
    2: "task_size",
    3: "task_color",
    4: "task_rot",
    5: "task_flip",
    6: "task_count",
    7: "task_inside",
    8: "task_contact",
    9: "task_sym_rot",
    10: "task_sym_mir",
    ### compositions
    11: "task_pos_pos_1",
    12: "task_pos_pos_2",
    13: "task_pos_count_2",
    14: "task_pos_count_1",
    15: "task_pos_pos_4",
    16: "task_pos_count_3",
    17: "task_inside_count_1",
    18: "task_count_count",
    19: "task_shape_shape",
    20: "task_shape_contact_2",
    21: "task_contact_contact_1",
    22: "task_inside_inside_1",
    23: "task_inside_inside_2",
    24: "task_pos_inside_3",
    25: "task_pos_inside_1",
    26: "task_pos_inside_2",
    27: "task_pos_inside_4",
    28: "task_rot_rot_1",
    29: "task_flip_flip_1",
    30: "task_rot_rot_3",
    31: "task_pos_pos_3",
    32: "task_pos_count_4",
    33: "task_size_size_1",
    34: "task_size_size_2",
    35: "task_size_size_3",
    36: "task_size_size_4",
    37: "task_size_size_5",
    38: "task_size_sym_1",
    39: "task_size_sym_2",
    40: "task_color_color_1",
    41: "task_color_color_2",
    42: "task_sym_sym_1",
    43: "task_sym_sym_2",
    44: "task_shape_contact_3",
    45: "task_shape_contact_4",
    46: "task_contact_contact_2",
    47: "task_pos_size_1",
    48: "task_pos_size_2",
    49: "task_pos_shape_1",
    50: "task_pos_shape_2",
    51: "task_pos_rot_1",
    52: "task_pos_rot_2",
    53: "task_pos_col_1",
    54: "task_pos_col_2",
    55: "task_pos_contact",
    56: "task_size_shape_1",
    57: "task_size_shape_2",
    58: "task_size_rot",
    59: "task_size_inside_1",
    60: "task_size_contact",
    61: "task_size_count_1",
    62: "task_size_count_2",
    63: "task_shape_color",
    64: "task_shape_color_2",
    65: "task_shape_color_3",
    66: "task_shape_inside",
    67: "task_shape_inside_1",
    68: "task_shape_count_1",
    69: "task_shape_count_2",
    70: "task_rot_color",
    71: "task_rot_inside_1",
    72: "task_rot_inside_2",
    73: "task_rot_count_1",
    74: "task_color_inside_1",
    75: "task_color_inside_2",
    76: "task_color_contact",
    77: "task_color_count_1",
    78: "task_color_count_2",
    79: "task_inside_contact",
    80: "task_contact_count_1",
    81: "task_contact_count_2",
    ##### new
    82: "task_size_color_1",
    83: "task_size_color_2",
    84: "task_color_sym_1",
    85: "task_color_sym_2",
    86: "task_shape_rot_1",
    87: "task_shape_contact_5",
    88: "task_rot_contact_1",
    89: "task_rot_contact_2",
    90: "task_inside_sym_mir",
    91: "task_flip_count_1",
    92: "task_flip_inside_1",
    93: "task_flip_inside_2",
    94: "task_flip_color_1",
    95: "task_shape_flip_1",
    96: "task_rot_flip_1",
    97: "task_size_flip_1",
    98: "task_pos_rot_3",
    99: "task_pos_flip_1",
    100: "task_pos_flip_2",
    101: "task_flip_contact_1",
    102: "task_flip_contact_2",    
}


class CVRT(Dataset):
    
    def __init__(self, base_folder, task, split='train', n_samples=-1, image_size=128, transform=None):
        super().__init__()

        self.base_folder = base_folder
        if task =='a':
            self.tasks = [v for _,v in TASKS.items()]
        else:
            self.tasks = [TASKS[int(t)] for t in task.split('-')]
        
        self.split = split
        if n_samples > 0:
            self.n_samples = n_samples
        elif split == 'train':
            self.n_samples = 10000
        elif split == 'val':
            self.n_samples = 500
        elif split == 'test':
            self.n_samples = 1000

        self.image_size = image_size

        self.transform = transform
        self.totensor = tvt.ToTensor()

    def __len__(self):
        return len(self.tasks) * self.n_samples * 4

    def __getitem__(self, idx):
        task_idx = idx // (self.n_samples*4)
        sample_idx = idx % (self.n_samples*4)
        
        i = sample_idx % 4
        sample_idx = sample_idx // 4

        sample_path = os.path.join(self.base_folder, self.tasks[task_idx], self.split, '{:05d}.png'.format(sample_idx))
        sample = Image.open(sample_path)
        _, height = sample.size
        pad = (height-self.image_size)//2
        
        return self.transform(sample.crop((i*height + pad, pad, (i+1) * height - pad, height-pad)))
        