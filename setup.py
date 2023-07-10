from setuptools import setup
from setuptools import find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='xaddpy',
    packages=find_packages(),
    version='0.1.16',
    license='MIT License',
    description='XADD package in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jihwan Jeong',
    author_email='jiihwan.jeong@gmail.com',
    url='https://github.com/jihwan-jeong/xaddpy',
    download_url="https://github.com/jihwan-jeong/xaddpy/archive/refs/tags/0.1.16.tar.gz"
)