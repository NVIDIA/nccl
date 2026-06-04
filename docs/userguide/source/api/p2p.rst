**************************************
Point To Point Communication Functions
**************************************

NCCL provides two types of point-to-point communication primitives: two-sided operations and one-sided operations.

Two-Sided Point-to-Point Operations
====================================

(Since NCCL 2.7) Two-sided point-to-point communication primitives need to be used when ranks need to send and
receive arbitrary data from each other, which cannot be expressed as a broadcast or allgather, i.e.
when all data sent and received is different. Both sender and receiver must explicitly participate.

ncclSend
--------

.. c:function:: ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream)

 Send data from ``sendbuff`` to rank ``peer``.

 Rank ``peer`` needs to call ncclRecv with the same ``datatype`` and the same ``count`` as this rank.

 This operation is blocking for the GPU. If multiple :c:func:`ncclSend` and :c:func:`ncclRecv` operations
 need to progress concurrently to complete, they must be fused within a :c:func:`ncclGroupStart`/
 :c:func:`ncclGroupEnd` section.

Related links: :ref:`point-to-point`.

ncclRecv
--------

.. c:function:: ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream)

 Receive data from rank ``peer`` into ``recvbuff``.

 Rank ``peer`` needs to call ncclSend with the same ``datatype`` and the same ``count`` as this rank.

 This operation is blocking for the GPU. If multiple :c:func:`ncclSend` and :c:func:`ncclRecv` operations
 need to progress concurrently to complete, they must be fused within a :c:func:`ncclGroupStart`/
 :c:func:`ncclGroupEnd` section.

Related links: :ref:`point-to-point`.

One-Sided Point-to-Point Operations (RMA)
==========================================

One-sided Remote Memory Access (RMA) operations enable ranks to directly access remote memory without
explicit participation from the target process. These operations require the target memory to be
pre-registered within a symmetric memory window using :c:func:`ncclCommWindowRegister`.

ncclPutSignal
-------------

.. c:function:: ncclResult_t ncclPutSignal(const void* localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream)

 Write data from ``localbuff`` to rank ``peer``'s registered memory window ``peerWin`` at offset ``peerWinOffset``
 and subsequently updating a remote signal.

 The target memory window ``peerWin`` must be registered using :c:func:`ncclCommWindowRegister`.

 The ``sigIdx`` is the signal index identifier for the operation. It must be set to 0 for now.

 The ``ctx`` is the context identifier for the operation. It must be set to 0 for now.

 The ``flags`` parameter is reserved for future use. It must be set to 0 for now.

 The return of :c:func:`ncclPutSignal` to the CPU thread indicates that the operation has been successfully enqueued to the CUDA stream.
 At the completion of :c:func:`ncclPutSignal` on the CUDA stream, the ``localbuff`` is safe to reuse or modify.
 When a signal is updated on the remote peer, it guarantees that the data from the corresponding :c:func:`ncclPutSignal` operation has been delivered to the remote memory.
 All prior :c:func:`ncclPutSignal` and :c:func:`ncclSignal` operations to the same peer and context have also completed their signal updates.

Related links: :ref:`point-to-point`.

ncclSignal
----------

.. c:function:: ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream)

 Send a signal to rank ``peer`` without transferring data.

 The ``sigIdx`` is the signal index identifier for the operation. It must be set to 0 for now.

 The ``ctx`` is the context identifier for the operation. It must be set to 0 for now.

 The ``flags`` parameter is reserved for future use. It must be set to 0 for now.

 When a signal is updated on the remote peer, all prior :c:func:`ncclPutSignal` and :c:func:`ncclSignal` operations
 to the same peer and context have also completed their signal updates.

Related links: :ref:`point-to-point`.

ncclWaitSignal
--------------

.. c:type:: ncclWaitSignalDesc_t

 Descriptor that specifies how many signal operations to wait for
 from a particular rank on a given signal index and context.

 .. c:member:: int opCnt

  Number of signal operations to wait for.

 .. c:member:: int peer

  Target peer to wait for signals from.

 .. c:member:: int sigIdx

  Signal index identifier. Must be set to 0 for now.

 .. c:member:: int ctx

  Context identifier. Must be set to 0 for now.

.. c:function:: ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream)

 Wait for signals as described in the signal descriptor array.

 The ``nDesc`` parameter specifies the number of signal descriptors in the ``signalDescs`` array.
 Each descriptor indicates how many signals (``opCnt``) to expect from a specific ``peer``
 on a particular signal index (``sigIdx``) and context (``ctx``).

 The return of :c:func:`ncclWaitSignal` to the CPU thread indicates that the operation has been successfully enqueued to the CUDA stream.
 At the completion of :c:func:`ncclWaitSignal` on the CUDA stream, all specified signal operations have been received and the corresponding data is visible in local memory.

Related links: :ref:`point-to-point`.
