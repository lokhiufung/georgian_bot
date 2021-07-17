
from distutils.core import setup

from setuptools import setup, find_packages


def required(requirements_file):
    required_packages = []
    with open(requirements_file, 'r') as f:
        for line in f:
            required_packages.append(line.strip())
    return required_packages

# make sure that you have install the correct version of pytorch
import torch
assert '1.7.1' in torch.__version__
    

setup(
    name='Friday',
    version='2.0a',
    description='Neural networks powered tools for building virtual assistant',
    author='Einstein Lok',
    author_email='lokhiufung@gmail.com',
    url='https://github.com/friday',
    packages=find_packages(),
    install_required=required('requirements.txt')
)

