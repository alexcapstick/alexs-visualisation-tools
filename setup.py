import os
from setuptools import setup, find_packages
import subprocess
import logging
from avt import __version__, __doc__, __author__, __title__, __author_email__

PACKAGE_NAME = 'avt'


setup(
    name=__title__,
    version=__version__,
    description=__doc__,
    author=__author__,
    author_email=__author_email__,
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