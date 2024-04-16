from setuptools import setup, find_packages

setup(
    name="STIsim",
    description='STI modelling toolbox built on the starsim platform',
    version="0.1",
    url="https://github.com/starsimhub/syphilis_analyses",
    packages=find_packages(),
    install_requires=[
        'starsim'
    ],
)
