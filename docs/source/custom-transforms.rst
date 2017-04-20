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
