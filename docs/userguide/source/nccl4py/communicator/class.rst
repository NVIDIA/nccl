.. py:currentmodule:: nccl.core

******************
Communicator Class
******************

.. autoclass:: Communicator
   :no-members:
   :special-members: __init__

Properties
==========

Identity
--------

.. autoattribute:: Communicator.ptr
.. autoattribute:: Communicator.is_valid
.. autoattribute:: Communicator.nranks
.. autoattribute:: Communicator.device
.. autoattribute:: Communicator.rank

Device-API capability
---------------------

These properties reflect the underlying NCCL :c:type:`ncclCommProperties_t`
structure.

.. autoattribute:: Communicator.cuda_dev
.. autoattribute:: Communicator.nvml_dev
.. autoattribute:: Communicator.device_api_support
.. autoattribute:: Communicator.multimem_support
.. autoattribute:: Communicator.gin_type
.. autoattribute:: Communicator.n_lsa_teams
.. autoattribute:: Communicator.host_rma_support
.. autoattribute:: Communicator.railed_gin_type
