
def install_dinov2():
    import subprocess
    import sys
    import shutil
    import os

    # Clone the repository
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/dinov2.git"])

    # Navigate to the cloned directory
    import os
    os.chdir("dinov2")

    # Install dependencies
    # subprocess.run([sys.executable, "-m", "pip", "install", "-r", "repip iquirements.txt"])

    # Install the package
    subprocess.run([sys.executable, "setup.py", "install"])
        
    # Remove the cloned dinov2 folder
    os.chdir("..")
    shutil.rmtree("dinov2")