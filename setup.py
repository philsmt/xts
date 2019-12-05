import os
import subprocess

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

extra_kwargs = {}

try:
    from Cython.Build import cythonize
except ImportError:
    pass
else:
    import numpy

    extra_kwargs['ext_modules'] = cythonize([
        Extension(
            'xts.math._signal_native', ['xts/math/_signal_native.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['-O2', '-march=native', '-frename-registers',
                                '-ftree-vectorize', '-ffast-math']
        ),
    ], build_dir='./build/sources', language_level=3)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

version_str = 'dev'

try:
    short_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stderr=subprocess.STDOUT
    )
except Exception:
    pass
else:
    if not short_hash.startswith(b'fatal'):
        version_str = short_hash.decode('ascii').strip()

setup(
    name = 'XTS',
    version = version_str,
    author = 'Philipp Schmidt',
    author_email = 'philipp.schmidt@xfel.eu',
    description = ('An extendable toolkit for data analysis of single-shot data, e.g. at FEL experiments'),
    license = 'GPL',
    packages = ['xts', 'xts.math'],
    long_description = read('README.md'),
    long_description_content_type='text/markdown',
    python_requires = '>=3.6',
    install_requires = ['numpy', 'scipy'],
    classifiers = [
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    **extra_kwargs
)
