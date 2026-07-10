.. _in-place-operations:

*******************
In-place Operations
*******************

Contrary to MPI, NCCL does not define a special "in-place" value to replace pointers. Instead, NCCL optimizes the case where the provided pointers are effectively "in place".

For ncclBroadcast, ncclReduce and ncclAllreduce functions, this means that passing ``sendBuff == recvBuff`` will perform in place operations,
storing final results at the same place as initial data was read from.

For ncclReduceScatter and ncclAllGather, in place operations are done when the per-rank pointer is located at the rank offset of the global buffer.
More precisely, these calls are considered in place : ::

  ncclReduceScatter(data, data+rank*recvcount, recvcount, datatype, op, comm, stream);
  ncclAllGather(data+rank*sendcount, data, sendcount, datatype, op, comm, stream);

