.. _point-to-point:

****************************
Point-to-point communication
****************************

Two-sided communication
========================

(Since NCCL 2.7)
Point-to-point communication can be used to express any communication pattern between ranks.
Any point-to-point communication needs two NCCL calls: a call to :c:func:`ncclSend` on one
rank and a corresponding :c:func:`ncclRecv` on the other rank, with the same count and data
type.

Multiple calls to :c:func:`ncclSend` and :c:func:`ncclRecv` targeting different peers
can be fused together with :c:func:`ncclGroupStart` and :c:func:`ncclGroupEnd` to form more
complex communication patterns such as one-to-all (scatter), all-to-one (gather),
all-to-all or communication with neighbors in an N-dimensional space.

Point-to-point calls within a group will be blocking until that group of calls completes,
but calls within a group can be seen as progressing independently, hence should never block
each other. It is therefore important to merge calls that need to progress concurrently to
avoid deadlocks. The only exception is point-to-point calls within a group targeting the
*same* peer, which are executed in order.

Below are a few examples of classic point-to-point communication patterns used by parallel
applications. NCCL semantics allow for all variants with different sizes,
datatypes, and buffers, per rank.

Sendrecv
--------

In MPI terms, a sendrecv operation is when two ranks exchange data, both sending and receiving
at the same time. This can be done by merging both ncclSend and ncclRecv calls into one:

.. code:: C

 ncclGroupStart();
 ncclSend(sendbuff, sendcount, sendtype, peer, comm, stream);
 ncclRecv(recvbuff, recvcount, recvtype, peer, comm, stream);
 ncclGroupEnd();

One-to-all (scatter)
--------------------

A one-to-all operation from a ``root`` rank can be expressed by merging all send and receive
operations in a group:

.. code:: C

 ncclGroupStart();
 if (rank == root) {
   for (int r=0; r<nranks; r++)
     ncclSend(sendbuff[r], size, type, r, comm, stream);
 }
 ncclRecv(recvbuff, size, type, root, comm, stream);
 ncclGroupEnd();

All-to-one (gather)
-------------------

Similarly, an all-to-one operation to a ``root`` rank would be implemented this way:

.. code:: C

 ncclGroupStart();
 if (rank == root) {
   for (int r=0; r<nranks; r++)
     ncclRecv(recvbuff[r], size, type, r, comm, stream);
 }
 ncclSend(sendbuff, size, type, root, comm, stream);
 ncclGroupEnd();

All-to-all
----------

An all-to-all operation would be a merged loop of send/recv operations
to/from all peers:

.. code:: C

 ncclGroupStart();
 for (int r=0; r<nranks; r++) {
   ncclSend(sendbuff[r], sendcount, sendtype, r, comm, stream);
   ncclRecv(recvbuff[r], recvcount, recvtype, r, comm, stream);
 }
 ncclGroupEnd();

Neighbor exchange
-----------------

Finally, exchanging data with neighbors in an N-dimensional space could be done
with:

.. code:: C

 ncclGroupStart();
 for (int d=0; d<ndims; d++) {
   ncclSend(sendbuff[d], sendcount, sendtype, next[d], comm, stream);
   ncclRecv(recvbuff[d], recvcount, recvtype, prev[d], comm, stream);
 }
 ncclGroupEnd();

One-sided communication
========================

(Since NCCL 2.29)
One-sided communication enables a rank to write data to remote memory using :c:func:`ncclPutSignal`
without requiring the target rank to issue a matching operation. The target memory must be pre-registered
using :c:func:`ncclCommWindowRegister`. Point-to-point synchronization can be achieved by having the
target rank call :c:func:`ncclWaitSignal` to wait for signals.

Multiple :c:func:`ncclPutSignal` calls can be grouped using :c:func:`ncclGroupStart` and
:c:func:`ncclGroupEnd`. Operations to different peers or contexts within a group may execute
concurrently and complete in any order. The completion of :c:func:`ncclGroupEnd` guarantees that
all operations in the group have achieved completion.
Operations to the same peer and context are executed in order: both data delivery and signal
updates on the remote peer follow the program order.

Below are a few examples of classic one-sided communication patterns used by parallel applications.

PutSignal and WaitSignal
------------------------

A ping-pong pattern using :c:func:`ncclPutSignal` and :c:func:`ncclWaitSignal`.
This example shows the full setup including memory allocation and window registration:

.. code:: C

 // Allocate symmetric memory for RMA operations
 void *sendbuff, *recvbuff;
 NCCLCHECK(ncclMemAlloc((void**)&sendbuff, size));
 NCCLCHECK(ncclMemAlloc((void**)&recvbuff, size));

 // Register buffers as symmetric windows
 ncclWindow_t sendWindow, recvWindow;
 NCCLCHECK(ncclCommWindowRegister(comm, sendbuff, size, &sendWindow, NCCL_WIN_COLL_SYMMETRIC));
 NCCLCHECK(ncclCommWindowRegister(comm, recvbuff, size, &recvWindow, NCCL_WIN_COLL_SYMMETRIC));

 int peer = (rank == 0) ? 1 : 0;
 ncclWaitSignalDesc_t waitDesc = {.opCnt = 1, .peer = peer, .sigIdx = 0, .ctx = ctx};

 if (rank == 0) {
   // Rank 0: wait then put
   NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));
   NCCLCHECK(ncclPutSignal(sendbuff, count, datatype, peer, recvWindow, 0,
                     0, 0, 0, comm, stream));
 } else {
   // Rank 1: put then wait
   NCCLCHECK(ncclPutSignal(sendbuff, count, datatype, peer, recvWindow, 0,
                     0, 0, 0, comm, stream));
   NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));
 }

 CUDACHECK(cudaStreamSynchronize(stream));

 // Cleanup
 NCCLCHECK(ncclCommWindowDeregister(comm, sendWindow));
 NCCLCHECK(ncclCommWindowDeregister(comm, recvWindow));
 NCCLCHECK(ncclMemFree(sendbuff));
 NCCLCHECK(ncclMemFree(recvbuff));


Barrier
-------

A barrier pattern using :c:func:`ncclSignal` and :c:func:`ncclWaitSignal`.
Each rank signals to all other ranks and waits for signals from all ranks:

.. code:: C

 ncclWaitSignalDesc_t *waitDescs = malloc(nranks * sizeof(ncclWaitSignalDesc_t));
 for (int r = 0; r < nranks; r++) {
   waitDescs[r].opCnt = 1;
   waitDescs[r].peer = r;
   waitDescs[r].sigIdx = 0;
   waitDescs[r].ctx = 0;
 }

 ncclGroupStart();
 for (int r = 0; r < nranks; r++) {
   ncclSignal(r, 0, 0, 0, comm, stream);
 }
 ncclGroupEnd();

 ncclWaitSignal(nranks, waitDescs, comm, stream);

All-to-all
----------

An all-to-all operation using :c:func:`ncclPutSignal`.
Each rank sends data to all other ranks and waits for signals from all ranks.
User needs to register the memory window for each peer using :c:func:`ncclCommWindowRegister` in advance.
User needs to guarantee the buffers are ready before calling :c:func:`ncclPutSignal`.
This could be done with the barrier shown above.

.. code:: C

 size_t offset[nranks];
 ncclWaitSignalDesc_t *waitDescs = malloc(nranks * sizeof(ncclWaitSignalDesc_t));
 for (int r = 0; r < nranks; r++) {
   offset[r] = r * count * wordSize(datatype);
   waitDescs[r].opCnt = 1;
   waitDescs[r].peer = r;
   waitDescs[r].sigIdx = 0;
   waitDescs[r].ctx = 0;
 }

 ncclGroupStart();
 for (int r = 0; r < nranks; r++) {
   ncclPutSignal(sendbuff[r], count, datatype, r, window, offset[r],
           0, 0, 0, comm, stream);
 }
 ncclGroupEnd();

 ncclWaitSignal(nranks, waitDescs, comm, stream);

