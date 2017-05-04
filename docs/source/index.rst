HiDi: Pipelines for Latent Factor Modeling
==========================================

HiDi is a library for high-dimensional latent factor modeling for collaborative
filtering applications.

Why HiDi?
---------

We created HiDi because modeling latent factors for collaborative filtering
applications is a work intensive process that involves many data
transformations, each of which requires special consideration to get a
good result. HiDi makes the process more simple by breaking work into small
steps, each of which can be executed in a pipeline.

The unit of work in HiDi is a Transformer. Transformers need only implement
one function, `transform`.

Ok, How Do I Use It?
--------------------

This will get you started.

.. literalinclude:: ../../hidi/examples/simple.py

Setup
-----

Requirements
~~~~~~~~~~~~

HiDi is tested against CPython 2.7, 3.4, 3.5, and 3.6. It may work with
different version of CPython.

Installation
~~~~~~~~~~~~

To install HiDi, simply run

.. code-block:: sh

   $ pip install hidi

API Documentation
-----------------

.. toctree::
   :maxdepth: 4

   index
   pipeline
   inout
   matrix
   clean
   forking
   custom-transforms
