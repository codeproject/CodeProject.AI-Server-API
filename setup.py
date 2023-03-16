from setuptools import setup, find_packages

VERSION = "0.2"

REQUIRES = ["requests"]

setup(
    name             = "codeproject-ai-api",
    version          = VERSION,
    url              = "https://github.com/codeproject/codeproject-ai-api",
    author           = "CodeProject",
    author_email     = "info@codeproject.com",
    description      = "Provides a simple SDK for the CodeProject.AI Server API",
    install_requires = REQUIRES,
    packages         = find_packages(),
    license          = "Apache License, Version 2.0",
    python_requires  = ">=3.7",
    classifiers      = [
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
