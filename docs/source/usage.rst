Usage
=====

.. _installation:

Installation
------------

TE2Rules package is available on PyPI and can be installed with pip:

.. code-block:: console

   $ pip install te2rules

Explaining Tree Ensemble Models
--------------------------------

To use TE2Rules, start with instantiating a ``te2rules.explainer.ModelExplainer`` with the 
tree ensemble model to be explained.

.. autoclass:: te2rules.explainer.ModelExplainer
   :noindex:

.. autofunction:: te2rules.explainer.ModelExplainer.__init__
   :noindex:

To explain, the tree ensemble model with a list of rules:

.. autofunction:: te2rules.explainer.ModelExplainer.explain

To evaluate the extracted rule list:

.. autofunction:: te2rules.explainer.ModelExplainer.get_fidelity

