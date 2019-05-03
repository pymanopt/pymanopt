from io import open
from os import path
from setuptools import setup, find_packages


with open(path.join(path.abspath(path.dirname(__file__)),
                    'VERSION')) as f:
    version = f.read().strip()


with open(path.join(path.abspath(path.dirname(__file__)),
                    'README.md'),
          encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pymanopt',
    version=version,
    description=("Toolbox for optimization on manifolds with support for "
                 "automatic differentiation"),
    url='https://pymanopt.github.io',
    author='Jamie Townsend, Niklas Koep and Sebastian Weichwald',
    license='BSD',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords=('optimization,manifold optimization,automatic differentiation,'
              'machine learning,numpy,scipy,theano,autograd,tensorflow'),
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.10', 'scipy>=0.17'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    data_files=[
        'CONTRIBUTING.md',
        'CONTRIBUTORS',
        'LICENSE',
        'MAINTAINERS',
        'README.md',
        'VERSION',
    ],
)
