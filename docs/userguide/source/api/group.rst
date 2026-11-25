***********
Group Calls
***********

Group primitives define the behavior of the current thread to avoid blocking. They can therefore be used from multiple threads independently.

Related links: :ref:`group-calls`.

ncclGroupStart
--------------

.. c:function:: ncclResult_t ncclGroupStart()

 Start a group call.

 All subsequent calls to NCCL until :c:func:`ncclGroupEnd` will not block due to inter-CPU synchronization.

ncclGroupEnd
------------

.. c:function:: ncclResult_t ncclGroupEnd()

 End a group call.

 Returns when all operations since :c:func:`ncclGroupStart` have been processed. This means the communication primitives
 have been enqueued to the provided streams, but are not necessarily complete.

 With nonblocking communicators, this function may return
 :c:enumerator:`ncclInProgress <ncclResult_t.ncclInProgress>` before processing is complete.
 In this case poll :c:func:`ncclCommGetAsyncError` on each communicator in the group until its *asyncError*
 output is no longer :c:enumerator:`ncclInProgress <ncclResult_t.ncclInProgress>`.
 All communicators must report :c:enumerator:`ncclSuccess <ncclResult_t.ncclSuccess>` before issuing further
 NCCL calls or using the affected CUDA streams. If a query fails or a communicator reports a terminal error,
 stop polling and follow the error-handling guidance in :c:func:`ncclCommGetAsyncError`.

 With blocking communicators, a group containing :c:func:`ncclCommInitRank` calls waits for all
 communicators to be initialized before :c:func:`ncclGroupEnd` returns.

ncclGroupSimulateEnd
--------------------

.. c:function:: ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo)

 Simulate a :c:func:`ncclGroupEnd` call and return NCCL's simulation info in a structure passed as an argument.
