import re 
from pathlib import Path 
import os
import subprocess as sp

from setuptools import setup, find_packages

FILE = Path(__file__).resolve()
PARENT = FILE.parent 
README = (PARENT / 'README.md').read_text(encoding='utf-8')

# def get_version():
#     file = PARENT / 'mm/__init__.py'
    
#     return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]


def get_version():
    file = PARENT / "mm/__init__.py"
    version = re.search(
        r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding="utf-8"), re.M
    )[1]

    try:
        version = (
            sp.check_output(["git", "describe", "--tags", "--abbrev=0"])
            .decode()
            .strip()[1:]
        )
        return os.environ.get(
            "PACKAGE_VERSION", version
        )  # Default version if git tag retrieval fails
    except Exception as err:
        print("Error:", err)
        return os.environ.get(
            "PACKAGE_VERSION", version
        )  # Default version if git tag retrieval fails


def parse_requirements(file_path: Path):
    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        
        if line and not line.startswith('#'):
            requirements.append(line.split("#")[0].strip())
    
    return requirements 

# setup(
#     name='mm',
#     version=get_version(),
#     python_requires='>=3.9',
#     description=('Template'),
#     long_description=README,
#     long_description_content_type='text/markdown',
#     # packages=['athena'] + [str(x) for x in Path('athena').rglob('*/') if x.is_dir() and '__' not in str(x)],
#     packages=find_packages(exclude=[]),
#     package_data={
#         '': ['*.yaml', '*.json'], },
#     include_package_data=True,
#     # install_requires=parse_requirements(PARENT / 'mm/segmentation/requirements.txt') + ["nvidia-ml-py3", "imgviz"],
#     # install_requires=["nvidia-ml-py3", "imgviz"],
# )

try:
    setup(
        name='mm',
        version='{{VERSION_PLACEHOLDER}}',
        python_requires='>=3.9',
        description=('Template for Python Athena'),
        long_description=README,
        long_description_content_type='text/markdown',
        # packages=['athena'] + [str(x) for x in Path('athena').rglob('*/') if x.is_dir() and '__' not in str(x)],
        packages=find_packages(exclude=[]),
        package_data={
            '': ['*.yaml', '*.json'], },
        include_package_data=True,
        install_requires=["nvidia-ml-py3", 'imgviz'],
    )
except:
    setup(
        name='mm',
        version=get_version(),
        python_requires='>=3.9',
        description=('Template for Python Athena'),
        long_description=README,
        long_description_content_type='text/markdown',
        # packages=['athena'] + [str(x) for x in Path('athena').rglob('*/') if x.is_dir() and '__' not in str(x)],
        packages=find_packages(exclude=[]),
        package_data={
            '': ['*.yaml', '*.json'], },
        include_package_data=True,
        # install_requires=["nvidia-ml-py3"],
        install_requires=["nvidia-ml-py3"],
    )