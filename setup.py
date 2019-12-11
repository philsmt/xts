
import os
import subprocess

from setuptools import setup
from setuptools.extension import Extension

extra_kwargs = {}

try:
    from Cython.Build import cythonize
except ImportError:
    pass
else:
    import numpy

    extra_kwargs['ext_modules'] = cythonize([
        Extension(
            'xts.math._vmi_native', ['xts/math/_vmi_native.pyx'],
            include_dirs=[numpy.get_include()],
            # Do not use -ffast-math for this one, needs further checking!
            extra_compile_args=['-O2', '-march=native', '-frename-registers',
                                '-ftree-vectorize', '-fopenmp'],
            extra_link_args=['-fopenmp']
        ),
        Extension(
            'xts.math._signal_native', ['xts/math/_signal_native.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['-O2', '-march=native', '-frename-registers',
                                '-ftree-vectorize', '-ffast-math']
        ),
    ], build_dir='./build/sources', language_level=3)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


try:
    short_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stderr=subprocess.STDOUT
    )
except Exception:
    version_str = 'dev'
else:
    if not short_hash.startswith(b'fatal'):
        version_str = short_hash.decode('ascii').strip()

setup(
    name='XTS',
    version=version_str,
    author='Philipp Schmidt',
    author_email='philipp.schmidt@xfel.eu',
    description='An extendable toolkit for data analysis of shot data, e.g. '
                'at FEL experiments',
    license='License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
    packages=['xts', 'xts.math'],
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
    ],
    **extra_kwargs
)
