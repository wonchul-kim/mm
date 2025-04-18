import yaml
from datetime import datetime
from pathlib import Path
import os
import os.path as osp 
import numpy as np

def add_params_to_args(args, params_file):
    with open(params_file) as yf:
        params = yaml.load(yf, Loader=yaml.FullLoader)

    for key, value in params.items():
        setattr(args, key, value)
        
        
def create_output_dirs(args, mode='train'):
    if mode == 'train':
        now = datetime.now()
        args.output_dir = osp.join(args.output_dir, f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}')     
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)
            print(f" - Created {args.output_dir}")
        
        args.val_dir = osp.join(args.output_dir, 'val')
        if not osp.exists(args.val_dir):
            os.mkdir(args.val_dir)
            print(f" - Created {args.val_dir}")
        
        args.debug_dir = osp.join(args.output_dir, 'debug')
        if not osp.exists(args.debug_dir):
            os.mkdir(args.debug_dir)
            print(f" - Created {args.debug_dir}")
        
        args.logs_dir = osp.join(args.output_dir, 'logs')
        if not osp.exists(args.logs_dir):
            os.mkdir(args.logs_dir)
            print(f" - Created {args.logs_dir}")
        
        args.weights_dir = osp.join(args.output_dir, 'weights')
        if not osp.exists(args.weights_dir):
            os.mkdir(args.weights_dir)
            print(f" - Created {args.weights_dir}")

    elif mode == 'test':
        args.output_dir = str(increment_path(args.output_dir))

    else:
        raise NotImplementedError(f"There is no such mode: {mode}")
        
def increment_path(path, exist_ok=False, sep="", mkdir=False):
    from glob import glob
    import re 
    
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        dirs = glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def numpy_converter(obj):
    if isinstance(obj, (np.float32, np.float64, np.int64, np.integer, np.floating)):
        return obj.item()  # convert to Python native type
    raise TypeError(f"Type {type(obj)} not serializable")
