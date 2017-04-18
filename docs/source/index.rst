HiDi: Pipelines for Embeddings
==============================

HiDi is a library for high-dimensional embedding generation for collaborative
filtering applications.

Why HiDi?
---------

We created HiDi because generating embeddings for collaborative filtering
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

Writing Custom Transforms
-------------------------

Writing a custom transform is simple and straighforward.
A transformer must only implement one function, :code:`transform`.
After initialization, transformers should be stateless so
they may be used in multiple pipelines, and each pipeline
can be executed many times. Keeping transformers stateless
also helps with memory consumption, which can become a
problem as the size of input grows.

Here is an example transform class implementation:

.. code-block:: python

  import hidi


  class TimesTwoTransform(object):
      def transform(self, inp, **kwargs):
          # Transform input
          return inp*2, kwargs

  pipeline = hidi.pipeline.Pipeline([
      ...,
      TimesTwoTransform(),
      ...
  ])

  pipeline.run()
