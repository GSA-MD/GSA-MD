#!/bin/env python

from setuptools import setup, find_packages

with open('./README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f.readlines() if line.strip()]

setup(
    name='gsa_md',
    version='0.1',
    maintainer='Francesco Massimo',
    maintainer_email='francesco.massimo@universite-paris-saclay.fr',
    #description='...',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    install_requires=install_requires,
    python_requires='>=3.8',
)