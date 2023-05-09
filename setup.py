import codecs
import glob
import os
import shutil

from setuptools import find_packages, setup

# Basic information
NAME = "MultiMachineMPICluster"
DESCRIPTION = "A Python library for using multiple machines in a cluster to train deep learning models."
VERSION = "0.0.1"
AUTHOR = "MultiMachineMPICluster authors"
EMAIL = "partha.ghosh@tuebingen.mpg.de"
LICENSE = "MIT"
REPOSITORY = "https://github.com/ParthaEth/MultiMachineMPICluster"
PACKAGE = "mmmc"

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Define the keywords
KEYWORDS = ["multi-gpu", "pytorch", "machine learning", "deep learning"]

# Define the classifiers
# See https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3.9",
]

# Important Paths
PROJECT = os.path.abspath(os.path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"
PKG_DESCRIBE = "README.md"

# Directories to ignore in find_packages
EXCLUDES = ()


# helper functions
def read(*parts):
    """
    returns contents of file
    """
    with codecs.open(os.path.join(PROJECT, *parts), "rb", "utf-8") as file:
        return file.read()


def get_requires(path=REQUIRE_PATH):
    """
    generates requirements from file path given as REQUIRE_PATH
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


# Define the configuration
CONFIG = {
    "name": NAME,
    "version": VERSION,
    "description": DESCRIPTION,
    "long_description": LONG_DESCRIPTION,
    "long_description_content_type": "text/markdown",
    "classifiers": CLASSIFIERS,
    "keywords": KEYWORDS,
    "license": LICENSE,
    "author": AUTHOR,
    "author_email": EMAIL,
    "url": REPOSITORY,
    "project_urls": {"Source": REPOSITORY},
    "packages": find_packages(
        where=PROJECT, include=["mmmc", "mmmc.*"], exclude=EXCLUDES
    ),
    "install_requires": list(get_requires()),
    "python_requires": "==3.9.*",
    "test_suite": "tests",
    "tests_require": ["pytest>=3"],
    "include_package_data": True,
}

if __name__ == "__main__":
    setup(**CONFIG)
