import sys

from codecs import open
from distutils.core import setup
from setuptools import find_packages


def get_install_requires():
    requires = [
        'scipy>=0.16.0',
        'tqdm>=4.0.0',
        'numpy>=1.8.0',
        'scikit-learn>=0.17.0',
        'pyvalid>=0.6',
        'pandas>=0.17.0',
        'nimfa>=1.1.0',
    ]

    if sys.version_info < (3, 2):
        requires.append('futures>=3.1.0')

    return requires


with open('README.rst', 'r', 'utf-8') as f:
    readme = f.read()


setup(
    name='HiDi',
    version='0.0.3',
    description='High-dimensional embedding generation library',
    long_description=readme,
    license='ALv2',
    author='Vevo Engineering',
    author_email='engineering@vevo.com',
    url='https://github.com/vevo/hidi',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=get_install_requires(),
    tests_require=[
        'nose==1.3.7',
        'Sphinx==1.5.5',
        'sphinx-autobuild==0.6.0',
        'tox==2.7.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
)
