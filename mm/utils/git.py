import subprocess
import sys
import shutil
import os
import time
import torch.distributed as dist


def install_by_clone(url, dir_name, no_deps=True, delete=True):
    base_dir = "/tmp"  # 또는 "/opt" / "/workspace"
    clone_path = os.path.join(base_dir, dir_name)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
        
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>> rank: ", rank)
    print("dist.is_available(): ", dist.is_available())
    print("dist.is_initialized(): ", dist.is_initialized())
    
    if rank == 0:

        print(f"[Rank {rank}] START to Clone: {url} into {clone_path}")
    
        # 기존 디렉토리 삭제 (이미 존재할 경우 대비)
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path)

        resp = subprocess.run(["git", "clone", url, clone_path], check=True)
        if resp.returncode != 0:
            raise RuntimeError(f"[Rank {rank}] Failed to clone the repository for {dir_name}")

        os.chdir(clone_path)
        # subprocess.run([sys.executable, "-m", "pip", "install", "-r", "repip iquirements.txt"])

        # subprocess.run([sys.executable, "setup.py", "install"])

        # env = os.environ.copy()
        # env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
        # subprocess.run([sys.executable, "setup.py", "install", "--single-version-externally-managed"], env=env, check=True)

        if no_deps:
            subprocess.run([sys.executable, "-m", "pip", "install", ".", "--no-deps"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "."], check=True)


        print(f"[Rank {rank}] Installation Completed !!!")
        
    if dist.is_available() and dist.is_initialized():
        print(f"[Rank {rank}] Waiting for Rank 0 to finish installation...")
        dist.barrier()
        time.sleep(2)
        

    if delete and rank == 0:
        os.chdir("/tmp")  # 안전한 경로로 이동 후 삭제
        shutil.rmtree(clone_path)
        print(f"[Rank {rank}] Deleted {clone_path}")


if __name__ == '__main__':
    install_by_clone(url="https://github.com/facebookresearch/sam2.git", dir_name='sam2')  
    