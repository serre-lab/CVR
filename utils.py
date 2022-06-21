import os
import sys
import yaml
import argparse

def parse_args(parser, argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args, unknownargs = parser.parse_known_args(argv)
    # args = parser.parse_args(argv)
    
    config_vars = {}
    
    if args.config is not None:
        with open(args.config, 'r') as stream:
            config_vars = yaml.load(stream, Loader=yaml.FullLoader)
            
        default_args = argparse.Namespace()
        default_args.__dict__.update(args.__dict__)
        default_args.__dict__.update(config_vars)

        new_keys = {}
        for k, v in args.__dict__.items():
            if '--'+k in argv or '-'+k in argv or (k not in default_args):
                new_keys[k] = v

        default_args.__dict__.update(new_keys)
        args = default_args
    
    return args

def save_config(cfg, path):
    with open(path, 'w') as cfg_file:
        yaml.dump(cfg, cfg_file)

def find_best_epoch(save_dir, version):
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    ckpt_folder = save_dir

    ckpt_files = os.listdir(ckpt_folder)  # list of strings
    ckpt_files = list(filter(lambda x: '.ckpt' in x, ckpt_files))
    epochs = [int(s.split('-')[0].split('=')[1]) for s in ckpt_files]
    best_file = ckpt_files[epochs.index(max(epochs))]
    
    return os.path.join(ckpt_folder, best_file) 


def process_results(results, task):
    global_avg = {k: r.mean() for k,r in results.items()}
    if task == 'a':
        tasks = list(range(103))
    elif task == 'elem':
        tasks = list(range(9))
    elif task == 'comp':
        tasks = list(range(9, 103))
    else:
        tasks = task.split('-')
    per_task = {}
    per_task_avg = {}

    if len(tasks) > 1:
        
        for k,r in results.items():
            k_res = r.reshape([len(tasks), -1])
            for i, t in enumerate(tasks):
                per_task['{}_t_{}'.format(k, t)] = k_res[i]
                per_task_avg['{}_t_{}'.format(k, t)] = k_res[i].mean()
    else:
        for k,v in results.items():
            per_task['{}_t_{}'.format(k, task)] = v
            per_task_avg['{}_t_{}'.format(k, task)] = v.mean()

    return global_avg, per_task, per_task_avg
