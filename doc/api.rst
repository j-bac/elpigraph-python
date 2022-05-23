.. automodule:: elpigraph

API
===

Import the package as::

   import elpigraph

Principal graphs
----------------

.. autosummary::
   :toctree: .

   computeElasticPrincipalCurve
   computeElasticPrincipalCircle
   computeElasticPrincipalTree
   generateInitialConfiguration

Graph editing
-------------
.. autosummary::
   :toctree: .

   addPath
   delPath
   findPaths
   ExtendLeaves
   CollapseBranches
   ShiftBranching
   fineTuneBR

Utils
-----
.. autosummary::
   :toctree: .

   utils.getProjection
   utils.getPseudotime
   utils.getWeights

Plotting
--------

.. autosummary::
   :toctree: .

   plot.PlotPG