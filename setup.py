from setuptools import setup
from setuptools import find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='xaddpy',
    packages=['xaddpy'],
    install_requires=['numpy', 'sympy', 'pulp'],
    version='v0.1.0',
    license='MIT License',
    description='XADD package in Python',
    long_description=long_description,
    author='Jihwan Jeong',
    author_email='jiihwan.jeong@gmail.com',
    url='https://github.com/jihwan-jeong/xaddpy',
    keywords=["xadd", "xadd python", "symbolic diagram"],
    download_url="https://github.com/jihwan-jeong/xaddpy/archive/refs/tags/v0.1.0.tar.gz"
)