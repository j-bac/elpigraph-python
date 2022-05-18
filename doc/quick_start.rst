Quick Start
===========

Principal graphs can be built in this way:

.. code-block:: python

    import elpigraph
    import numpy as np

    #generate data : np.array (n_points x n_dim).
    data = np.random.random((1000,10))

    #build principal graph with chosen topology
    pg_curve = elpigraph.computeElasticPrincipalCurve(data,NumNodes=50)
    pg_circle = elpigraph.computeElasticPrincipalCircle(data,NumNodes=50)
    pg_tree = elpigraph.computeElasticPrincipalTree(data,NumNodes=50)

    #retrieve graph (node positions and edges)
    pg_curve[0]['NodePositions'],pg_curve[0]['Edges']
