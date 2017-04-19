HiDi: Pipelines for Embeddings
==============================

HiDi is a library for high-dimensional embedding generation for collaborative
filtering applications.

Read the `full documentation <http://hidi.rtfd.io/>`_.

How Do I Use It?
----------------

This will get you started.

.. code-block:: python

   from hidi import inout, clean, matrix, pipeline


   # CSV file with link_id and item_id columns
   in_files = ['hidi/examples/data/user-item.csv']

   # File to write output data to
   outfile = 'embeddings.csv'

   transforms = [
       inout.ReadTransform(in_files),      # Read data from disk
       clean.DedupeTransform(),            # Dedupe it
       matrix.SparseTransform(),           # Make a sparse user*item matrix
       matrix.SimilarityTransform(),       # To item*item similarity matrix
       matrix.SVDTransform(),              # Perform SVD dimensionality reduction
       matrix.ItemsMatrixToDFTransform(),  # Make a DataFrame with an index
       inout.WriteTransform(outfile)       # Write results to csv
   ]

   pl = pipeline.Pipeline(transforms)
   pl.run()


Setup
-----

Requirements
~~~~~~~~~~~~

HiDi is tested against CPython 2.7, 3.4, 3.5, and 3.6. It may work with
different version of CPython.

Installation
~~~~~~~~~~~~

To install HiDi, simply run

.. code-block:: bash

   $ pip install hidi


Run the Tests
-------------

.. code-block:: bash

   $ pip install tox
   $ tox
