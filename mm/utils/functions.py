import yaml
from datetime import datetime
import os
import os.path as osp 

def add_params_to_args(args, params_file):
    with open(params_file) as yf:
        params = yaml.load(yf, Loader=yaml.FullLoader)

    for key, value in params.items():
        setattr(args, key, value)
        
        
def create_output_dirs(output_dir):
    now = datetime.now()
    output_dir = osp.join(output_dir, f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}')     
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        print(f" - Created {output_dir}")
        
    val_dir = osp.join(output_dir, 'val')
    if not osp.exists(val_dir):
        os.mkdir(val_dir)
        print(f" - Created {val_dir}")
    
    debug_dir = osp.join(output_dir, 'debug')
    if not osp.exists(debug_dir):
        os.mkdir(debug_dir)
        print(f" - Created {debug_dir}")
    
    logs_dir = osp.join(output_dir, 'logs')
    if not osp.exists(logs_dir):
        os.mkdir(logs_dir)
        print(f" - Created {logs_dir}")
    
    weights_dir = osp.join(output_dir, 'weights')
    if not osp.exists(weights_dir):
        os.mkdir(weights_dir)
        print(f" - Created {weights_dir}")
        
    return output_dir
    