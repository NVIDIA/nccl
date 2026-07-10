*************
Thread Safety
*************

NCCL primitives are generally not thread-safe, however, they are reentrant. Under multi-thread environment, it is not allowed
to issue NCCL operations to a single communicator in parallel with multiple threads; it is not safe to issue NCCL operations
in parallel to independent communicators located on the same device with multiple threads (see :ref:`multi-thread-concurrent-usage`).
If the child communicator shares the resources with the parent communicator (i.e., :ref:`ncclconfig` by `splitShare`), it is not
allowed to issue NCCL operations to the child and parent communicators in parallel.

It is safe to operate a communicator from multiple threads as long as users can guarantee only one thread
operates the communicator at a time. However, for any grouped NCCL operations, users need to ensure
only one thread issues all the operations in the group.

For example, the following code provides a simple thread-safe example where threads are executed in sequence and only one thread
is accessing the communicator at a time.

.. code:: C

  Thread 0:
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    cudaSetDevice(0);
    ncclCommInitRankConfig(&comm, nranks, id, rank, &config);
    ncclGroupStart();
    ncclAllReduce(sendbuff0, recvbuff0, count0, datatype, redOp, comm, stream);
    ncclAllReduce(sendbuff1, recvbuff1, count1, datatype, redOp, comm, stream);
    ncclGroupEnd();
    thread_exit();
  Thread 1:
    ncclResult_t state = ncclSuccess;
    // wait for previous issued allreduce ops by Thread 0
    do {
      ncclCommGetAsyncError(comm, &state);
    } while (state == ncclInProgress);
    assert(state == ncclSuccess);
    ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, redOp, comm, stream);
    do {
      ncclCommGetAsyncError(comm, &state);
    } while (state == ncclInProgress);
    assert(state == ncclSuccess);

It is also valid to issue grouped NCCL operations from one thread and poll the status of each NCCL
communicator with one thread as shown in the following code.

.. code:: C

  Thread 0:
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    ncclGroupStart();
    for (int i = 0; i < nGpus; i++) {
      cudaSetDevice(i);
      ncclCommInitRankConfig(&comms[i], nranks, id, ranks[i], &config);
    }
    ncclGroupEnd();
  Thread 0/1/2/3:
    ncclResult_t state = ncclSuccess;
    // wait for previous issued init ops by Thread 0
    do {
      ncclCommGetAsyncError(comms[thread_id], &state);
    } while (state == ncclInProgress);
    assert(state == ncclSuccess);
    ncclAllReduce(sendbuff, recvbuff, count, datatype, redOp, comms[thread_id], stream);
    do {
      ncclCommGetAsyncError(comms[thread_id], &state);
    } while (state == ncclInProgress);
    assert(state == ncclSuccess);
