.. py:currentmodule:: nccl.core

********************************
Collective Communication Methods
********************************

Methods on :py:class:`Communicator` for collective communication. See
:doc:`../../api/colls` for the corresponding C API.

Each collective below takes an optional :py:class:`NCCLCollConfig` as
``config`` to customize that one call; see :doc:`../configuration`.

allreduce
---------
.. automethod:: Communicator.allreduce

broadcast
---------
.. automethod:: Communicator.broadcast

reduce
------
.. automethod:: Communicator.reduce

allgather
---------
.. automethod:: Communicator.allgather

reduce_scatter
--------------
.. automethod:: Communicator.reduce_scatter

alltoall
--------
.. automethod:: Communicator.alltoall

gather
------
.. automethod:: Communicator.gather

scatter
-------
.. automethod:: Communicator.scatter

create_pre_mul_sum
------------------
.. automethod:: Communicator.create_pre_mul_sum
