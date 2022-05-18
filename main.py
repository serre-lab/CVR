import os
import sys
# import yaml sys
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


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []
        self.all_keys = []

    def on_validation_epoch_end(self, trainer, pl_module):
        
        each_me = {}
        for k,v in trainer.callback_metrics.items():
            each_me[k] = v.item()
            if k not in self.all_keys:
                self.all_keys.append(k)

        self.metrics.append(each_me)

    def get_all(self):

        all_metrics = {}
        for k in self.all_keys:
            all_metrics[k] = []

        for m in self.metrics[1:]:
            for k in self.all_keys:
                v = m[k] if k in m else np.nan
                all_metrics[k].append(v)
        
        return all_metrics

def cli_main():

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")

    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--exp_dir', type=str, default='../experiments/', help='experiment output directory')
    parser.add_argument('--path_db', type=str, default='../dbs', help='experiment database path')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--resume_training', action='store_true', help='resume training from checkpoint training')
    parser.add_argument('--model', type=str, default='', help='self supervised training method')
    parser.add_argument('--dataset', type=str, default='', help='dataset to use for training')
    parser.add_argument('--ckpt_period', type=int, default=3, help='save checkpoints every')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for resuming training or finetuning')
    parser.add_argument('--finetune', type=int, default=0, help='0 = no finetuning')
    parser.add_argument('--freeze_pretrained', type=int, default=0, help='0 = no freezing')
    parser.add_argument('--early_stopping', type=int, default=0, help='0 = no early stopping')
    parser.add_argument('--refresh_rate', type=int, default=10, help='progress bar refresh rate')
    parser.add_argument('--es_patience', type=int, default=40, help='early stopping patience')
    
    

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

    fit_kwargs = {}
    # save config
    if args.resume_training:
        
        ckpt = list(filter(lambda x: '.ckpt' in x, os.listdir(args.exp_dir)))[-1]
        ckpt = os.path.join(args.exp_dir, ckpt)
            
        print('resuming from checkpoint', ckpt)
        # args.__dict__.update({'resume_from_checkpoint': latest_ckpt})
        fit_kwargs['ckpt_path'] = ckpt

    else:
        if args.checkpoint != '' and args.finetune == 1:
            ckpt = args.checkpoint
            if '.ckpt' not in ckpt:
                ckpt = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt)))[0]
                ckpt = os.path.join(args.checkpoint, ckpt)

            model.load_finetune_weights(ckpt)

        elif args.checkpoint != '':
            ckpt = args.checkpoint
            
            model.load_backbone_weights(ckpt)

        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.path_db, exist_ok=True)
        save_config(args.__dict__, os.path.join(args.exp_dir, 'config.yaml'))

    if args.freeze_pretrained == 1:
        model.freeze_pretrained()

    # if args.resume_training:
    #     latest_ckpt = find_best_epoch(args.exp_dir, 0)
    #     print('resuming from checkpoint', latest_ckpt)
    #     args.__dict__.update({'resume_from_checkpoint': latest_ckpt})
        
    # training
    logger = TensorBoardLogger(args.exp_dir, default_hp_metric=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.exp_dir, save_top_k=1, mode='max', monitor='metrics/val_acc', every_n_epochs=args.ckpt_period, save_last=True) 
    callbacks = [model_checkpoint]
    if args.early_stopping!=0:
        early_stopping = pl.callbacks.EarlyStopping(monitor='metrics/val_acc', mode='max', patience=args.es_patience, stopping_threshold=0.99) #0.99
        callbacks.append(early_stopping)
    callbacks.append(TQDMProgressBar(refresh_rate=args.refresh_rate))
    metrics_callback = MetricsCallback()
    callbacks.append(metrics_callback)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
    
    # logger.log_hyperparams(model.hparams, metrics={'hp/'+k : v for k, v in model_type._hp_log_metrics.items()})

    trainer.fit(model, datamodule, **fit_kwargs)

    # testing
    # best_ckpt = model_checkpoint.best_model_path if model_checkpoint.best_model_path!="" else model_checkpoint.last_model_path
    
    best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)
    
    trainer.test(model=best_model, datamodule=datamodule)
    train_result = best_model.test_results

    global_avg, per_task, per_task_avg = process_results(train_result, args.task)    

    metrics = metrics_callback.get_all()


    best_val_acc = np.nanmax(metrics['metrics/val_acc'])
    best_epoch = (np.nanargmax(metrics['metrics/val_acc'])+1) * args.ckpt_period

    # saving results
    logger.log_hyperparams(best_model.hparams, metrics={'hp/'+k : v for k, v in global_avg.items()})
    logger.save()

    df = pd.DataFrame()
    output_dict = {
        '0_train': 0,
        '0_exp_name': args.exp_name,
        '0_exp_dir': args.exp_dir,
        '0_model': args.model,
        '0_seed': args.seed,
        '0_dataset': args.dataset,
        '0_checkpoint': args.checkpoint,
        '0_finetune': args.finetune,
        '0_freeze_pretrained': args.freeze_pretrained,
        '1_task': args.task,
        '1_n_samples': args.n_samples,
        '2_val_acc': best_val_acc,
        '2_best_epoch': best_epoch,
        '3_max_epochs': args.max_epochs,
        '3_backbone': args.backbone,
        '3_batch_size': args.batch_size,
        '3_lr': args.lr,
        '3_wd': args.wd,
    }
    
    output_dict.update({'2_'+k:v for k,v in global_avg.items()})
    output_dict.update({'5_'+k:v for k,v in per_task_avg.items()})
    
    results_save_path = os.path.join(args.exp_dir, 'results.npy')
    np.save(results_save_path, {'global_avg': global_avg, 'per_task_avg': per_task_avg, 'per_task': per_task, 'metrics': metrics})
    # hparams_dict = {'hp_'+k:v for k,v in best_model.hparams.items()}
    # output_dict.update(hparams_dict)

    df = df.append(output_dict, ignore_index=True)
    df.to_csv(os.path.join(args.path_db, args.exp_name + '_db.csv'))

if __name__ == '__main__':
    print(os.getpid())

    cli_main()
