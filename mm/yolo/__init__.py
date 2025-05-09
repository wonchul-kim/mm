import os
import os.path as osp
import pkg_resources
import shutil
import subprocess as sp
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parent 

def init_mmyolo():
    try:
        mmyolo_path = pkg_resources.get_distribution('mmyolo').location + '/mmyolo'
        assert osp.exists(mmyolo_path), ValueError(f'There is no mmyolo library: {mmyolo_path}')
    except pkg_resources.DistributionNotFound:
        print('mmyolo is not installed')
        sp.run(["mim", "install", "mmyolo"], check=True)
        print('INSTALLED mmyolo')

        mmyolo_path = pkg_resources.get_distribution('mmyolo').location + '/mmyolo'
        assert osp.exists(mmyolo_path), ValueError(f'There is no mmyolo library: {mmyolo_path}')
        
    new_init_path = str(ROOT / 'utils/init/__init__.py')
    assert osp.exists(new_init_path), ValueError(f'There is no new mmyolo init file: {new_init_path}')
    print(f"FOUND new_init_path for mmyolo: {new_init_path}")

    existing_init_path = os.path.join(mmyolo_path, '__init__.py')
    assert osp.exists(existing_init_path), ValueError(f'There is no existing mmyolo init file: {existing_init_path}')
    shutil.copy(new_init_path, existing_init_path)
    print(f"COPIED new_init_path into original mmyolo __init__: {new_init_path}")

