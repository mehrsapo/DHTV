*DHTV* is a framework to learn CPWL functions in an interpretable manner.

In this repository, we aim to:

* Reproduce of the results of the following paper:

  * `Delaunay-Triangulation-Based Learning with Hessian Total-Variation Regularization <https://arxiv.org/pdf/2208.07787.pdf>`_;

* Facilate learning with CPWL functions as the mentioned papers.


.. contents:: **Table of Contents**
    :depth: 2

Installation
============

To install the package, we first create an environment with python 3.7 (or greater):

.. code-block:: bash

    >> conda create -n DHTV python==3.9.7
    >> conda activate DHTV

Developper Install
------------------

.. code-block:: bash

   >> git clone https://github.com/mehrsapo/DHTV.git
   >> cd <repository_dir>/
   >> conda install -n DHTV ipykernel --update-deps --force-reinstall
   >> pip install --upgrade -r requirements.txt

Usage
=====
Learning example
-------------------
Here, we show an example on how to learn with DHTV:

.. code-block:: python

    tri = MyDelaunay(X_train, y_train)
    tri.construct_forward_matrix()
    tri.construct_regularization_matrix()


Reproducing results
-------------------

The paper reults are available in notebooks IV.A, IV.B and IV.C. 

Developers
==========

*DHTV* is developed by the `Biomedical Imaging Group <http://bigwww.epfl.ch/>`_,
`École Polytéchnique Fédérale de Lausanne <https://www.epfl.ch/en/>`_, Switzerland.

References
==========

.. [Pourya2022]  <https://arxiv.org/pdf/2208.07787.pdf>

Acknowledgements
================

This work was supported in part by the European Research Council (ERC Project FunLearn) under Grant 101020573 and in part by the Swiss National Science Foundation, Grant 200020 184646/1.