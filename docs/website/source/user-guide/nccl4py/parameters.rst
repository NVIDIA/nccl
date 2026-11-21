.. py:currentmodule:: nccl.core

**********
Parameters
**********

Read access to NCCL's tunable parameters. See :doc:`../env` for the meaning
of each parameter and how to set it.

.. code-block:: python

    import nccl.core

    nccl.core.params["NCCL_DEBUG"]     # value of a single parameter
    list(nccl.core.params)             # every parameter name
    nccl.core.dump_params()            # print all parameters to stdout

params
======

.. py:data:: params

   Read-only :py:class:`~collections.abc.Mapping` of NCCL parameter names to
   their current values, backed by :c:func:`ncclParamGetParameter` and
   :c:func:`ncclParamGetAllParameterKeys`. Values are returned as strings.
   Lookups are live: each access queries NCCL rather than a cached snapshot.

dump_params
===========
.. autofunction:: dump_params
