*DHTV* is a framework to learn CPWL functions in an interpretable manner.

In this repository, we aim to:

* Solve the regression problem with CPWL functions.

* Reproduce of the results of the following paper:
    * `Delaunay-Triangulation-Based Learning with Hessian Total-Variation Regularization <https://arxiv.org/pdf/2208.07787.pdf>`_.



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
First we need to build the DHTV model and compute forward (H) and regularization (L) operators:
.. code-block:: python

    tri = MyDelaunay(X, y)  # X: input variables, y: target values
    tri.construct_forward_matrix() # constructing H
    tri.construct_regularization_matrix() # constructing L

Then we solve the learning task: 
.. code-block:: python

    dhtv_sol, _ = double_fista(tri.data_values, tri.H, tri.L, tri.lip_H, tri.lip_L, lmbda, n_iter1, n_iter2, device='cuda:0')

We can use this values to predict the model values: 
.. code-block:: python

    dhtv_predict = tri.evaluate(X, dhtv_sol.cpu().numpy())

In the 2 dimensional case, we can also plot the model using:
.. code-block:: python
    tri.update_values(dhtv_sol.cpu().numpy())
    plot_with_gradient_map(tri, 0.5, 1, 1, 1)

See for more details <https://github.com/mehrsapo/DHTV/blob/main/intp_metric.ipynb>. 
    
Reproducing results
-------------------

The paper reults are available in notebooks IV.A, IV.B and IV.C, main_compare.py is resposible for creating the loaded data in IV.C. 

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