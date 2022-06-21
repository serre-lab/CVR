import os
import argparse
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms as tvt

from datasets.base_datamodules import DataModuleBase

from PIL import Image

class CVRTDataModule(DataModuleBase):

    def __init__(
        self,
        data_dir,
        task,
        train_transform,
        test_transform,
        n_samples,
        test_set,
        num_workers,
        batch_size,
        # image_size,
        **kwargs,
    ):

        super(CVRTDataModule, self).__init__(   num_workers,
                                                batch_size,
        )

        transform = self._default_transforms()

        self.train_set = CVRT(data_dir, task, split='train', n_samples=n_samples, image_size=128, transform=transform)
        self.val_set = CVRT(data_dir, task, split='val', n_samples=-1, image_size=128, transform=transform)
        if test_set == 'gen':
            self.test_set = CVRT(data_dir, task, split='test_gen', n_samples=-1, image_size=128, transform=transform)
        else:
            self.test_set = CVRT(data_dir, task, split='test', n_samples=-1, image_size=128, transform=transform)

        # self.setup()

    def get_train_labels(self):
        return self.train_set.targets

    def _default_transforms(self):
        transforms = tvt.Compose([
            tvt.Normalize((0.9, 0.9, 0.9), (0.1, 0.1, 0.1)),
        ])
        return transforms
    
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
        parser.add_argument('--test_set', type=str, default='')

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
    ### compositions
    9: "task_sym_rot",
    10: "task_sym_mir",
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


# Dataset

class CVRT(Dataset):
    
    def __init__(self, base_folder, task, split='train', n_samples=-1, image_size=128, transform=None):
        super().__init__()

        self.base_folder = base_folder
        if task =='a':
            self.tasks = [v for _,v in TASKS.items()]
        elif task=='elem':
            self.tasks = [TASKS[i] for i in range(9)]
        elif task=='comp':
            self.tasks = [TASKS[i] for i in range(9, len(TASKS))]
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
        elif split == 'test_gen':
            self.n_samples = 1000

        self.image_size = image_size

        self.transform = transform
        self.totensor = tvt.ToTensor()

    def __len__(self):
        return len(self.tasks) * self.n_samples

    def __getitem__(self, idx):
        task_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples
        
        sample_path = os.path.join(self.base_folder, self.tasks[task_idx], self.split, '{:05d}.png'.format(sample_idx))
        sample = Image.open(sample_path)
        
        sample = self.totensor(sample)
        im_size = sample.shape[1]
        pad = im_size - self.image_size
        
        sample = sample.reshape([3, im_size, 4, im_size]).permute([2,0,1,3])[:, :, pad//2:-pad//2, pad//2:-pad//2]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, task_idx

