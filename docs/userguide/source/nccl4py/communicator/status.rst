.. py:currentmodule:: nccl.core

***************************
Status and Utility Methods
***************************

Methods on :py:class:`Communicator` for resource cleanup and error/status
queries.

close_all_resources
===================
.. automethod:: Communicator.close_all_resources

get_last_error
==============
.. automethod:: Communicator.get_last_error

get_async_error
===============
.. automethod:: Communicator.get_async_error

get_mem_stat
============
.. automethod:: Communicator.get_mem_stat

NcclCommMemStat
===============
.. autoclass:: NcclCommMemStat
   :members:
   :exclude-members: GpuMemSuspend, GpuMemSuspended, GpuMemPersist, GpuMemTotal

get_error_string
================

Module-level helper to render an NCCL result code as a human-readable string.

.. autofunction:: get_error_string
