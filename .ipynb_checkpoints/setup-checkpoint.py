# setup.py
from setuptools import setup, find_packages

setup(
    name="octagon_analysis",                 # Package name
    version="0.1",                    # Version number
    packages=['parse_data'],         # Automatically find all packages
    install_requires=['numpy', 'pandas'],   # List your dependencies here
    author="Tom Hagley",               # Author details
    description="functions to handle octagon data parsing and preprocessing",  # Package description
    classifiers=[                     # Optional classifiers for metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)