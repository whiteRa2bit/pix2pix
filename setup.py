from setuptools import setup, find_packages

import pix_pix

with open("README.md", 'r') as f:
    long_description = f.read()

requirements = [
    "torch==1.4.0",
    "numpy==1.18.1",
    "matplotlib==3.2.1",
    "wandb==0.10.11",
    "tqdm==4.48.2",
    "loguru==0.5.3",
]

setup(
    name='pix_pix',
    version=pix_pix.__version__,
    description='Image-to-Image Translation with Conditional Adversarial Networks',
    license="MIT",
    long_description=long_description,
    author='Pavel Fakanov',
    author_email='pavel.fakanov@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
