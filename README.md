Description
===========

This package provides a Python implementation of the ElPiGraph algorithm with multi-cpu and gpu support. A
self-contained description of the algorithm is available
[here](https://github.com/auranic/Elastic-principal-graphs/blob/master/ElPiGraph_Methods.pdf)
or on this [arXiv paper](https://arxiv.org/abs/1804.07580)

It replicates the [R implementation](https://github.com/Albluca/ElPiGraph.R),
coded by [Luca Albergante](https://github.com/Albluca) and should return exactly the same results. Please open an issue if you do  notice different output. Differences between the two versions are detailed in [differences.md](differences.md). This package extends initial work by [Louis Faure](https://github.com/LouisFaure/ElPiGraph.P) and [Alexis Martin](https://github.com/AlexiMartin/ElPiGraph.P).

A native MATLAB implementation of the algorithm (coded by [Andrei
Zinovyev](https://github.com/auranic/) and [Evgeny
Mirkes](https://github.com/Mirkes)) is also
[available](https://github.com/auranic/Elastic-principal-graphs)

Citation
========

When using this package, please cite our paper:

Albergante L, Mirkes E, Bac J, Chen H, Martin A, Faure L, Barillot E, Pinello L, Gorban AN, Zinovyev A. [Robust and scalable learning of complex intrinsic dataset geometry via ElPiGraph.](https://www.mdpi.com/1099-4300/22/3/296) Entropy 2020;22(3):296.

Requirements
============

This code was tested with Python 3.7.1, and requires the following packages:
- pandas
- scipy
- numba
- numpy
- python-igraph
- scikit-learn
- plotnine

In addition, to enable respectively multi-cpu, gpu support:
- multiprocessing
- cupy
https://docs-cupy.chainer.org/en/stable/install.html#

The requirements.txt file provides the versions this package has been tested with

Installation
====================
```bash
git clone https://github.com/j-bac/elpigraph-python.git
cd elpigraph-python
pip install .
```
or

```bash
pip install git+https://github.com/j-bac/elpigraph-python.git
```
