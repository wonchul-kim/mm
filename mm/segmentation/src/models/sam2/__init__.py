
def install_sam2():
    import subprocess
    import sys
    import shutil
    import os

    # Clone the repository
    resp = subprocess.run(["git", "clone", "https://github.com/facebookresearch/sam2.git"], check=True)

    if resp.returncode != 0:
        raise RuntimeError(f"Failed to clone the repository for sam2")

    import os
    os.chdir("sam2")

    # subprocess.run([sys.executable, "-m", "pip", "install", "-r", "repip iquirements.txt"])

    # subprocess.run([sys.executable, "setup.py", "install"])

    # env = os.environ.copy()
    # env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    # subprocess.run([sys.executable, "setup.py", "install", "--single-version-externally-managed"], env=env, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", ".", "--no-deps"], check=True)

    os.chdir("..")
    shutil.rmtree("sam2")