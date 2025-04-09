import yaml
from datetime import datetime
import os
import os.path as osp 

def add_params_to_args(args, params_file):
    with open(params_file) as yf:
        params = yaml.load(yf, Loader=yaml.FullLoader)

    for key, value in params.items():
        setattr(args, key, value)
        
        
def create_output_dirs(args):
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
        
    