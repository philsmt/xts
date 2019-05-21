import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'XTS',
    version = '0.0.1',
    author = 'Philipp Schmidt',
    author_email = 'phil.smt@gmail.com',
    description = ('An extendable toolkit for data analysis of single-shot data, e.g. at FEL experiments'),
    license = 'GPL',
    packages = ['xts',],
    long_description = read('README'),
    python_requires = '>=3.6',
    classifiers = [
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
)