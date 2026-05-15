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
