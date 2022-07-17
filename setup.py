from setuptools import setup
from setuptools import find_packages

setup(
    name='xaddpy',
    packages=find_packages(),
    install_requires=['numpy'],
    version='0.1',
    license='MIT License',
    description='XADD package in Python that handles bilinear expressions',
    long_description=open('README.md').read(),
)