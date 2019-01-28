=============
BayesPowerlaw
=============

*Written by Kristina Grigaityte.*

.. image:: tweet_powerlaw.png
   :height: 260px
   :width: 380px
.. image:: tweet_posterior.png
   :height: 260px
   :width: 380px


BayesPowerlaw is a Python package that fits a single or a mixture of power law distributions to data using a Bayesian inference approach. Posterior distributions of parameters are numerically determined by Markov chain Monte Carlo sampling. In addition, the package provides capability for 
- power law simulations
- data plotting
- maximum likelihood estimation

Installation
------------

BayesPowerlaw can be installed from
`PyPI <https://pypi.python.org/pypi/BayesPowerlaw>`_ using the pip package manager (version 9.0.0 or higher). At the command line::

    pip install BayesPowerlaw

The code for BayesPowerlaw is open source and available on
`GitHub <https://github.com/AtwalLab/BayesPowerlaw>`_.


Quick Start
-----------

To make the figures shown above, type this from within Python::

   import BayesPowerlaw as bp
   bp.demo()

Resources
---------

.. toctree::

   tutorial
   documentation

Contact
-------

For technical assistance or to report bugs, please 
contact `Kristina Grigaityte <kgrigait@cshl.edu>`_.

For general correspondence, please 
contact `Gurinder (Mickey) Atwal <atwal@cshl.edu>`_.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`