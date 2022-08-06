import os
import re
import runpy
from itertools import chain

from setuptools import find_packages, setup


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONAL_DEPENDENCIES = ("autograd", "tensorflow", "torch")


def parse_requirements_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f.read().splitlines()]


if __name__ == "__main__":
    requirements = parse_requirements_file("requirements/base.txt")

    install_requires = []
    optional_dependencies = {}
    for requirement in requirements:
        # We manually separate hard from optional dependencies.
        if any(
            requirement.startswith(optional_dependency)
            for optional_dependency in OPTIONAL_DEPENDENCIES
        ):
            match = re.match(r"([A-Za-z0-9\-_]+).*", requirement)
            if match is None:
                continue
            package = match.group(1)
            optional_dependencies[package] = [requirement]
        else:
            install_requires.append(requirement)

    dev_requirements = parse_requirements_file("requirements/dev.txt")
    extras_require = {"test": dev_requirements, **optional_dependencies}
    extras_require["all"] = list(chain(*extras_require.values()))

    pymanopt_version = runpy.run_path(
        os.path.join(BASE_DIR, "pymanopt", "_version.py")
    )

    with open(os.path.join(BASE_DIR, "README.md")) as f:
        long_description = f.read()

    setup(
        name="pymanopt",
        version=pymanopt_version["__version__"],
        description=(
            "Toolbox for optimization on manifolds with support for "
            "automatic differentiation"
        ),
        url="https://pymanopt.org",
        author="Jamie Townsend, Niklas Koep and Sebastian Weichwald",
        license="BSD",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        keywords=(
            "optimization,manifold optimization,"
            "automatic differentiation,machine learning,numpy,scipy,"
            "autograd,tensorflow"
        ),
        packages=find_packages(exclude=["tests"]),
        install_requires=install_requires,
        extras_require=extras_require,
        long_description=long_description,
        long_description_content_type="text/markdown",
        data_files=[
            "CONTRIBUTING.md",
            "CONTRIBUTORS",
            "LICENSE",
            "MAINTAINERS",
            "README.md",
        ],
    )
