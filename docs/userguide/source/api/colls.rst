**********************************
Collective Communication Functions
**********************************


The following NCCL APIs provide some commonly used collective operations.

ncclAllReduce
-------------

.. c:function:: ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)

 Reduces data arrays of length ``count`` in ``sendbuff`` using the ``op`` operation and leaves identical copies of the result in each ``recvbuff``.

 In-place operation will happen if ``sendbuff == recvbuff``.

Related links: :ref:`allreduce`.


ncclBroadcast
-------------

.. c:function:: ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)

 Copies ``count`` elements from ``sendbuff`` on the ``root`` rank to all ranks' ``recvbuff``.
 ``sendbuff`` is only used on rank ``root`` and ignored for other ranks.

 In-place operation will happen if ``sendbuff == recvbuff``.


.. c:function:: ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)

 Legacy in-place version of ``ncclBroadcast`` in a similar fashion to MPI_Bcast. A call to ::

  ncclBcast(buff, count, datatype, root, comm, stream)

 is equivalent to ::

  ncclBroadcast(buff, buff, count, datatype, root, comm, stream)

Related links: :ref:`broadcast`

ncclReduce
----------

.. c:function:: ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)

 Reduce data arrays of length ``count`` in ``sendbuff`` into ``recvbuff`` on the ``root`` rank using the ``op`` operation.
 ``recvbuff`` is only used on rank ``root`` and ignored for other ranks.

 In-place operation will happen if ``sendbuff == recvbuff``.

Related links: :ref:`reduce`.

ncclAllGather
-------------

.. c:function:: ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)

 Gathers ``sendcount`` values from all GPUs and leaves identical copies of the result in each ``recvbuff``, receiving data from rank ``i`` at offset ``i*sendcount``.

 Note: This assumes the receive count is equal to ``nranks*sendcount``, which means that ``recvbuff`` should have a size of at least ``nranks*sendcount`` elements.

 In-place operation will happen if ``sendbuff == recvbuff + rank * sendcount``.

Related links: :ref:`allgather`, :ref:`in-place-operations`.

ncclReduceScatter
-----------------

.. c:function:: ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)

 Reduce data in ``sendbuff`` from all GPUs using the ``op`` operation and leave the reduced result scattered over the devices so that the ``recvbuff`` on
 rank ``i`` will contain the i-th block of the result.

 Note:  This assumes the send count is equal to ``nranks*recvcount``, which means that ``sendbuff`` should have a size of at least ``nranks*recvcount`` elements.

 In-place operation will happen if ``recvbuff == sendbuff + rank * recvcount``.

Related links: :ref:`reducescatter`, :ref:`in-place-operations`.

ncclAlltoAll
------------

.. c:function:: ncclResult_t  ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)

 Each rank sends ``count`` values to all other ranks and receives ``count`` values from all other ranks. Data to send to destination rank ``j`` is taken from ``sendbuff+j*count`` and data received from source rank ``i`` is placed at ``recvbuff+i*count``.

 Note: This assumes both the total send and receive count is equal to ``nranks*count``, which means that ``sendbuff`` and ``recvbuff`` should have a size of at least ``nranks*count`` elements.

 In-place operation is currently not supported.

Related links: :ref:`alltoall`.

ncclGather
----------

.. c:function:: ncclResult_t  ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)

 Each rank sends ``count`` elements from ``sendbuff`` to the ``root`` rank. On the ``root`` rank, data from rank ``i`` is placed at ``recvbuff + i*count``. On non-root ranks, ``recvbuff`` is not used.

 Note: This assumes the receive count is equal to ``nranks*count``, which means that ``recvbuff`` should have a size of at least ``nranks*count`` elements.

 In-place operation will happen if ``sendbuff == recvbuff + root * count``.

Related links: :ref:`gather`.

ncclScatter
-----------

.. c:function:: ncclResult_t  ncclScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)

 Each rank receives ``count`` elements from the ``root`` rank. On the ``root`` rank, ``count`` elements from ``sendbuff + i*count`` are sent to rank ``i``. On non-root ranks, ``sendbuff`` is not used.

 Note: This assumes the send count is equal to ``nranks*count``, which means that ``sendbuff`` should have a size of at least ``nranks*count`` elements.

 In-place operation will happen if ``recvbuff == sendbuff + root * count``.

Related links: :ref:`scatter`.
