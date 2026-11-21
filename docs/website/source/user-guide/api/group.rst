***********
Group Calls
***********

Group primitives define the behavior of the current thread to avoid blocking.
They can therefore be used from multiple threads independently.

Related links: :ref:`group-calls`.

ncclGroupStart
--------------

.. c:function:: ncclResult_t ncclGroupStart()

 Start a group call.

 All subsequent calls to NCCL until :c:func:`ncclGroupEnd` will not block due to inter-CPU
 synchronization.  Work within the group does not begin processing until the
 final :c:func:`ncclGroupEnd` call.  While submitting work within the group,
 :c:enumerator:`ncclSuccess <ncclResult_t.ncclSuccess>` only indicates the group will include the operation, but it does
 not indicate completion or that stream operations have been enqueued. Blocking
 and non-blocking communicators cannot submit operations within the same group.

 Internally, operations within a group are not guaranteed to be issued in the
 same order they were written.  Collectives on the same communicator within a
 group may be joined.  Data dependencies should not exist between operations in
 a group.  Grouping must be consistent across peers.  Mixing communicator
 management functions (initialization, finalization, window registration) with
 communication functions is not allowed.  See
 :ref:`group-operation-ordering`.

ncclGroupEnd
------------

.. c:function:: ncclResult_t ncclGroupEnd()

 End a group.  Work is submitted only if this is the outermost group.

 For a group including standard (blocking) communicators, :c:func:`ncclGroupEnd` returns
 :c:enumerator:`ncclSuccess <ncclResult_t.ncclSuccess>` when all communication operations since :c:func:`ncclGroupStart` have been
 submitted to the respective streams (but not necessarily completed), or all
 communicator management functions have been completed.

 For a group of non-blocking communicators, :c:func:`ncclGroupEnd` returns
 :c:enumerator:`ncclInProgress <ncclResult_t.ncclInProgress>`, and additional work cannot be submitted to the communicators
 until :c:func:`ncclCommGetAsyncError` indicates the communicator reached
 :c:enumerator:`ncclSuccess <ncclResult_t.ncclSuccess>`, indicating all work was submitted or completed as described in
 the blocking case.  If a query fails or a communicator reports a terminal error,
 stop polling and follow the error-handling guidance in :c:func:`ncclCommGetAsyncError`.

 Groups may be nested.  If this is not the outermost group, :c:func:`ncclGroupEnd` returns
 :c:enumerator:`ncclSuccess <ncclResult_t.ncclSuccess>` but no work has been submitted.

ncclGroupSimulateEnd
--------------------

.. c:function:: ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo)

 Simulate a :c:func:`ncclGroupEnd` call and return NCCL's simulation info in a structure
 passed as an argument, then discard the group's operations.  Use this to
 estimate the time a group of operations will take to execute.  If this is not
 the outermost group, :c:enumerator:`ncclSuccess <ncclResult_t.ncclSuccess>` is returned but *estimatedTime* will show a
 number less than 0 (no estimation is done).
