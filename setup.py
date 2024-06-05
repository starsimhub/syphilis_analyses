from setuptools import setup, find_packages

setup(
    name="STIsim",
    description='STI modelling toolbox built on the Starsim platform',
    version="0.0.2",
    url="https://github.com/starsimhub/stisim",
    packages=find_packages(),
    install_requires=[
        'starsim',
        'optuna',
    ],
)
