Forking Module
--------------

HiDi's forking module exposes functionality for concurrent pipelines.
Forking us done with ordinary Transforms that take lists of pipelines upon
initialization.

.. automodule:: hidi.forking
   :show-inheritance:
   :members:


Example
~~~~~~~

Here is an example of using a :code:`ProcessForkTransform`:

.. literalinclude:: ../../hidi/examples/forking.py
