
def install_by_clone(url, dir_name, no_deps=True, delete=True):
    import subprocess
    import sys
    import shutil
    import os

    # Clone the repository
    resp = subprocess.run(["git", "clone", url], check=True)

    if resp.returncode != 0:
        raise RuntimeError(f"Failed to clone the repository for {dir_name}")

    import os
    os.chdir(f"{dir_name}")

    # subprocess.run([sys.executable, "-m", "pip", "install", "-r", "repip iquirements.txt"])

    # subprocess.run([sys.executable, "setup.py", "install"])

    # env = os.environ.copy()
    # env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    # subprocess.run([sys.executable, "setup.py", "install", "--single-version-externally-managed"], env=env, check=True)
    if no_deps:
        subprocess.run([sys.executable, "-m", "pip", "install", ".", "--no-deps"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "."], check=True)


    if delete:
        os.chdir("..")
        shutil.rmtree(f"{dir_name}")
    
    
    