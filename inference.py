import os
import sys
# import yaml
import argparse

# from typing import Union
# from warnings import warn

import numpy as np
import pandas as pd
# from scipy import stats

import pytorch_lightning as pl
import torch
# import torch.nn.functional as F
# from torch import nn
import torchvision

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
# from pytorch_lightning.callbacks import Callback

import modules
import datasets

from utils import parse_args, save_config, find_best_epoch, process_results

def cli_main():

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--load_checkpoint', action='store_true', help='resume training from checkpoint training')
    

    args = parse_args(parser, argv)
    
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    
    # model args
    model_type = vars(modules)[args.model] 
    parser = model_type.add_model_specific_args(parser)
    
    # dataset args
    dataset_type = vars(datasets)[args.dataset]
    parser = dataset_type.add_dataset_specific_args(parser)

    args = parse_args(parser, argv)


    # dataset args
    dataset_type = vars(datasets)[args.dataset]

    # initializing the dataset and model
    datamodule = dataset_type(**args.__dict__)

    
    ###################################################################################################
    # checkpoint loading and setup

    if args.load_checkpoint:
        
        ckpt = list(filter(lambda x: '.ckpt' in x, os.listdir(args.exp_dir)))[0]
        ckpt = os.path.join(args.exp_dir, ckpt)
        
        # latest_ckpt = find_best_epoch(args.exp_dir, 0)
        print('Loading checkpoint', ckpt)
        # args.__dict__.update({'resume_from_checkpoint': latest_ckpt})
        model = model_type.load_from_checkpoint(ckpt)
    else:
        model = model_type(**args.__dict__)


    run_test = True
    save_db = True

    if run_test:

        logger = TensorBoardLogger(args.exp_dir, default_hp_metric=False)
        # model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.exp_dir, save_top_k=3, mode='min', monitor='metrics/'+args.track_metric, period=args.ckpt_period)     
        
        trainer = pl.Trainer.from_argparse_args(args)

        trainer.test(model=model, datamodule=datamodule)
        train_result = model.test_results
        global_avg, per_task, per_task_avg = process_results(train_result, args.task)

        log_metrics = {'hp/'+k : v for k, v in global_avg.items()}

        print(log_metrics)

    print('model device', model.device)

        
    if run_test:
        logger.log_hyperparams(model.hparams, metrics=log_metrics)
        logger.save()    

    ###################################################################################################
    # saving results
    
    if save_db:
                
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
            '3_max_epochs': args.max_epochs,
            '3_backbone': args.backbone,
            '3_batch_size': args.batch_size,
            '3_lr': args.lr,
            '3_wd': args.wd,
        }
        
        
        output_dict.update({'2_'+k:v for k,v in global_avg.items()})
        output_dict.update({'5_'+k:v for k,v in per_task_avg.items()})

        #### merges files if file exists
        results_save_path = os.path.join(args.exp_dir, 'results.npy')
        if os.path.exists(results_save_path):
            results_file = np.load(results_save_path, allow_pickle=True).item()
            for k, v in per_task.items():
                results_file['per_task'][k] = v
            for k, v in per_task_avg.items():
                results_file['per_task_avg'][k] = v
        else:
            results_file = {'global_avg': global_avg, 'per_task_avg': per_task_avg, 'per_task': per_task}
        
        np.save(results_save_path, results_file)

        db_file = os.path.join(args.path_db, args.exp_name + '_db.csv')
        if os.path.exists(db_file):
            df = pd.read_csv(db_file, index_col=0)
        else:
            df = pd.DataFrame()
        
        df = df.append(output_dict, ignore_index=True)
        df.to_csv(db_file)

if __name__ == '__main__':
    print(os.getpid())
    
    cli_main()

