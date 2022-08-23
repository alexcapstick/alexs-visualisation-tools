import os
from setuptools import setup, find_packages
import subprocess
import logging

PACKAGE_NAME = 'avt'


setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    description="Some visualisation tools that I've found helpful!",
    author='Alexander Capstick',
    author_email='',
    packages=find_packages(),
    long_description=open('README.txt').read(),
    install_requires=[
                        "numpy>=1.22",
                        "pandas>=1.4",
                        "matplotlib>=3.5",
                        "seaborn>=0.11.2",
                        "tqdm>=4.64",
    ]
)