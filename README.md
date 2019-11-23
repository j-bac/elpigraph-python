# ElPiGraph.Python
Python implementation of the ElPiGraph algorithm with multi-cpu and gpu support

Description
===========

This package provides a Python implementation of the ElPiGraph algorithm. A
self-contained description of the algorithm is available
[here](https://github.com/auranic/Elastic-principal-graphs/blob/master/ElPiGraph_Methods.pdf)
or on this [arXiv paper](https://arxiv.org/abs/1804.07580)

It replicates the [R implementation](https://github.com/Albluca/ElPiGraph.R),
coded by [Luca Albergante](https://github.com/Albluca) and should return exactly the same results. Please open an issue if you notice different output. Missing functions and differences between the two versions are detailed in [differences.md](differences.md). This package extends initial work by [Louis Faure](https://github.com/LouisFaure/ElPiGraph.P) and [Alexis Martin](https://github.com/AlexiMartin/ElPiGraph.P).

A native MATLAB implementation of the algorithm (coded by [Andrei
Zinovyev](https://github.com/auranic/) and [Evgeny
Mirkes](https://github.com/Mirkes)) is also
[available](https://github.com/auranic/Elastic-principal-graphs)



Citation
========

When using this package, please cite our preprint:

Albergante, L.  et al . Robust and Scalable  Learning of Data Manifold with Complex Topologies via ElPiGraph.
arXiv: [1804.07580](https://arxiv.org/abs/1804.07580) (2018)

Requirements
============

This code was tested with Python 3.7.1, and requires the following packages:
- pandas
- scipy
- numba
- numpy
- python_igraph
- scikit_learn
- plotnine

And to enable GPU support :
- cupy

The requirements.txt file provides the versions this package has been tested with

Installation & Usage
====================
```bash
pip install git+https://github.com/j-bac/ElPiGraph.Python.git
```
