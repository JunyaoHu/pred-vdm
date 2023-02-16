from setuptools import setup, find_packages

setup(
    name='pred-vdm',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)