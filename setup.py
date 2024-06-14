import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'stisim', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.rst'), "r") as f:
    long_description = f.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]


setup(
    name="STIsim",
    description='STI modelling toolbox built on the Starsim platform',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    keywords=["agent-based model", "simulation", "disease", "epidemiology"],
    version=version,
    url="https://github.com/starsimhub/stisim",
    packages=find_packages(),
    platforms=["OS Independent"],
    include_package_data=True,
    classifiers=CLASSIFIERS,
    install_requires=[
        'starsim',
        'optuna',
    ],
)
