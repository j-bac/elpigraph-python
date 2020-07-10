#####################################
Quick Start with elpigraph-python
#####################################

``elpigraph-python`` provides Python implementations of scikit-learn compatible estimators. 

Installation
===================================================

To install with pip run::

    $ pip install git+https://github.com/j-bac/elpigraph-python.git

To install from source run::

    $ git clone https://github.com/j-bac/elpigraph-python
    $ cd elpigraph-python
    $ pip install .

Basic usage
===================================================

ElPiGraph can be used in this way:

.. code-block:: python

    import elpigraph
    import numpy as np

    #generate data : np.array (n_points x n_dim).
    data = np.random.random((1000,10))
    #fit principal tree with 50 nodes
    tree = elpigraph.computeElasticPrincipalTree(data,50)
    