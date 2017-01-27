from setuptools import setup, find_packages


setup(
    name='pymanopt',
    version='0.2.2',
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
)
