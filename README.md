[![Documentation Status](https://readthedocs.org/projects/elpigraph-python/badge/?version=latest)](https://elpigraph-python.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/j-bac/elpigraph-python.svg?branch=master)](https://travis-ci.com/j-bac/elpigraph-python)
[![Build status](https://ci.appveyor.com/api/projects/status/hvnmxujbjvthfxn4/branch/master?svg=true)](https://ci.appveyor.com/project/j-bac/elpigraph-python/branch/master)
[![CircleCI](https://circleci.com/gh/j-bac/elpigraph-python/tree/master.svg?style=shield)](https://circleci.com/gh/j-bac/elpigraph-python/tree/master)
[![codecov](https://codecov.io/gh/j-bac/elpigraph-python/branch/master/graph/badge.svg)](https://codecov.io/gh/j-bac/elpigraph-python)
[![GitHub license](https://img.shields.io/github/license/j-bac/elpigraph-python)](https://github.com/j-bac/elpigraph-python/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI:10.3390/e22030296](https://img.shields.io/badge/DOI-10.3390%2Fe22030296-blue)](https://doi.org/10.3390/e22030296)

Description
===========

This package provides a Python implementation of the ElPiGraph algorithm with cpu and gpu support. A
self-contained description of the algorithm is available
[here](https://github.com/auranic/Elastic-principal-graphs/blob/master/ElPiGraph_Methods.pdf)
or in the [paper](https://www.mdpi.com/1099-4300/22/3/296)

It replicates the [R implementation](https://github.com/j-bac/ElPiGraph.R),
coded by [Luca Albergante](https://github.com/Albluca) and should return exactly the same results. Please open an issue if you do  notice different output. Differences between the two versions are detailed in [differences.md](differences.md). This package extends initial work by [Louis Faure](https://github.com/LouisFaure/ElPiGraph.P) and [Alexis Martin](https://github.com/AlexiMartin/ElPiGraph.P).

A native MATLAB implementation of the algorithm (coded by [Andrei
Zinovyev](https://github.com/auranic/) and [Evgeny
Mirkes](https://github.com/Mirkes)) is also
[available](https://github.com/auranic/Elastic-principal-graphs)

Citation
========

When using this package, please cite our [paper](https://www.mdpi.com/1099-4300/22/3/296):

Albergante, L.  et al . Robust and Scalable Learning of Complex Intrinsic Dataset Geometry via ElPiGraph (2020)

Requirements
============

This code was tested with Python 3.7.1, and requires the following packages:
- pandas
- scipy
- numba
- numpy
- python-igraph
- scikit-learn

In addition, to enable gpu support:
- cupy
https://docs-cupy.chainer.org/en/stable/install.html#

The requirements.txt file provides the versions this package has been tested with

Installation
====================
```bash
git clone https://github.com/j-bac/elpigraph-python.git
cd elpigraph
pip install .
```
or

```bash
pip install git+https://github.com/j-bac/elpigraph-python.git
```
