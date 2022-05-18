import os
import sys
# import yaml sys
import math
import argparse
import copy

# from typing import Union
# from warnings import warn
import numpy as np
import pandas as pd

import pytorch_lightning as pl
# import torch
# import torch.nn.functional as F
# from torch import nn

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import TQDMProgressBar


import modules
import datasets

from utils import parse_args, save_config, find_best_epoch, process_results

from distutils.util import strtobool

# scheduler
# momentum
# validation
# mixed precision
# sync bn


class MocoMCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self, moco_m):
        super().__init__()
        self.moco_m = moco_m

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        m = 1. - 0.5 * (1. + math.cos(math.pi * trainer.global_step / trainer.max_epochs * pl_module.n_epoch_steps)) * (1. - self.moco_m)
        # print('global step, moco_m', trainer.global_step, trainer.max_epochs, pl_module.n_epoch_steps, trainer.max_epochs * pl_module.n_epoch_steps, m)
        pl_module.moco_m = m

    # def adjust_moco_momentum(epoch, args):
    #     """Adjust moco momentum based on current epoch"""
    #     m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    #     return m




def cli_main():

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")

    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--exp_dir', type=str, default='../experiments/', help='experiment output directory')
    parser.add_argument('--path_db', type=str, default='../dbs', help='experiment database path')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--resume_training', action='store_true', help='resume training from checkpoint training')
    parser.add_argument('--model', type=str, default='CNN_VAE', help='self supervised training method')
    parser.add_argument('--dataset', type=str, default='SVHNSupDataModule', help='dataset to use for training')
    parser.add_argument('--ckpt_period', type=int, default=3, help='save checkpoints every')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for resuming training or finetuning')

    # parser.add_argument("--flag", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)


    args = parse_args(parser, argv)

    if args.seed is not None:
        pl.seed_everything(args.seed)
    
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    
    # model args
    model_type = vars(modules)[args.model] 
    parser = model_type.add_model_specific_args(parser)
    
    # dataset args
    dataset_type = vars(datasets)[args.dataset]
    parser = dataset_type.add_dataset_specific_args(parser)

    args = parse_args(parser, argv)

    # initializing the dataset and model
    datamodule = dataset_type(**args.__dict__)
    model = model_type(**args.__dict__)

    print(model.hparams)

    # save config
    if args.resume_training:
        latest_ckpt = find_best_epoch(args.exp_dir, 0)
        print('resuming from checkpoint', latest_ckpt)
        args.__dict__.update({'resume_from_checkpoint': latest_ckpt})

    else:
        

        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.path_db, exist_ok=True)
        save_config(args.__dict__, os.path.join(args.exp_dir, 'config.yaml'))


    # training
    logger = TensorBoardLogger(args.exp_dir, default_hp_metric=False)
    # model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.exp_dir, save_top_k=1, mode='min', monitor='metrics/train_loss', every_n_epochs=args.ckpt_period, save_last=True) 
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.exp_dir, save_top_k=1, every_n_epochs=args.ckpt_period, save_last=True) 
    callbacks = [model_checkpoint]
    # if args.early_stopping!=0:
    #     early_stopping = pl.callbacks.EarlyStopping(monitor='metrics/val_acc', mode='max', patience=40, stopping_threshold=1.01) #0.99
    #     callbacks.append(early_stopping)
    callbacks.append(TQDMProgressBar(refresh_rate=10))
    if args.moco_m_cos:
        mocom_callback = MocoMCallback(args.moco_m)
    callbacks.append(mocom_callback)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks) #, find_unused_parameters=False
    
    trainer.fit(model, datamodule)

    # testing
    # best_ckpt = model_checkpoint.best_model_path if model_checkpoint.best_model_path!="" else model_checkpoint.last_model_path
    
    # best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)
    
    # trainer.test(model=best_model, datamodule=datamodule)


    df = pd.DataFrame()
    output_dict = {
        '0_train': 0,
        '0_exp_name': args.exp_name,
        '0_exp_dir': args.exp_dir,
        '0_model': args.model,
        '0_seed': args.seed,
        '0_dataset': args.dataset,
        '1_task': args.task,
        '1_n_samples': args.n_samples,
        '3_max_epochs': args.max_epochs,
        '3_arch': args.arch,
        # '3_backbone': args.backbone,
        '3_batch_size': args.batch_size,
        '3_lr': args.lr,
        '3_weight_decay': args.weight_decay,
    }
    
    df = df.append(output_dict, ignore_index=True)
    df.to_csv(os.path.join(args.path_db, args.exp_name + '_db.csv'))

if __name__ == '__main__':
    print(os.getpid())

    cli_main()
