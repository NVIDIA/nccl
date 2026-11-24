.. _group-calls:

***********
Group Calls
***********

Group functions (:c:func:`ncclGroupStart`/:c:func:`ncclGroupEnd`) can be used to merge multiple calls into one. This is needed for
three purposes: managing multiple GPUs from one thread (to avoid deadlocks), aggregating communication operations
to improve performance, or merging multiple send/receive point-to-point operations (see :ref:`point-to-point`
section). All three usages can be combined together, with one exception: calls to :c:func:`ncclCommInitRank`
cannot be merged with others.

Management Of Multiple GPUs From One Thread
-------------------------------------------

When a single thread is managing multiple devices, group semantics must be used.
This is because every NCCL call may have to block, waiting for other threads/ranks to arrive, before effectively posting the NCCL operation on the given stream. Hence, a simple loop on multiple devices like shown below could block on the first call waiting for the other ones:

.. warning::
   We do not recommend using CUDA graph capture when managing multiple GPUs from one thread. In some cases ``cudaGraphLaunch`` may block, preventing the launch across all GPUs. See :ref:`using-nccl-with-cuda-graphs` for details.

.. code:: C

 for (int i=0; i<nLocalDevs; i++) {
   ncclAllReduce(..., comm[i], stream[i]);
 }

To define that these calls are part of the same collective operation, :c:func:`ncclGroupStart` and :c:func:`ncclGroupEnd` should be used:

.. code:: C

  ncclGroupStart();
  for (int i=0; i<nLocalDevs; i++) {
    ncclAllReduce(..., comm[i], stream[i]);
  }
  ncclGroupEnd();

This will tell NCCL to treat all calls between :c:func:`ncclGroupStart` and :c:func:`ncclGroupEnd` as a single call to many devices.

Caution: When called inside a group, stream operations (like :c:func:`ncclAllReduce`) can return without having enqueued the
operation on the stream. Stream operations like ``cudaStreamSynchronize`` can therefore be called only after :c:func:`ncclGroupEnd`
returns.

Group calls must also be used to create a communicator when one thread manages more than one device:

.. code:: C

  ncclGroupStart();
  for (int i=0; i<nLocalDevs; i++) {
    cudaSetDevice(device[i]);
    ncclCommInitRank(comms+i, nranks, commId, rank[i]);
  }
  ncclGroupEnd();


Note: Contrary to NCCL 1.x, there is no need to set the CUDA device before every NCCL communication call within a group,
but it is still needed when calling :c:func:`ncclCommInitRank` within a group.

Related links:

* :c:func:`ncclGroupStart`
* :c:func:`ncclGroupEnd`

Aggregated Operations (2.2 and later)
-------------------------------------

The group semantics can also be used to have multiple collective operations performed within a single NCCL launch. This
is useful for reducing the launch overhead, in other words, latency, as it only occurs once for multiple operations.
Initialization calls for different devices can be grouped, as shown above, but cannot be aggregated
into a communication kernel or mixed with communication calls in the same group.

Aggregation of collective operations can be done simply by having multiple calls to NCCL within a :c:func:`ncclGroupStart` /
:c:func:`ncclGroupEnd` section.

In the following example, we launch one broadcast and two allReduce operations together as a single NCCL launch.

.. code:: C

 ncclGroupStart();
 ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm, stream);
 ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, ncclSum, comm, stream);
 ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, ncclSum, comm, stream);
 ncclGroupEnd();

It is permitted to combine aggregation with multi-GPU launch and use different communicators in a group launch
as shown in the Management Of Multiple GPUs From One Thread topic. When combining multi-GPU launch and aggregation,
:c:func:`ncclGroupStart` and :c:func:`ncclGroupEnd` can be either used once or at each level. The following example groups the allReduce
operations from different layers and on multiple CUDA devices:

.. code:: C

 ncclGroupStart();
 for (int i=0; i<nlayers; i++) {
   ncclGroupStart();
   for (int g=0; g<ngpus; g++) {
     ncclAllReduce(sendbuffs[g]+offsets[i], recvbuffs[g]+offsets[i], counts[i], datatype[i], ncclSum, comms[g], streams[g]);
   }
   ncclGroupEnd();
 }
 ncclGroupEnd();

Note: The NCCL operation will only be started as a whole during the last call to :c:func:`ncclGroupEnd`. The :c:func:`ncclGroupStart` and
:c:func:`ncclGroupEnd` calls within the for loop are not necessary and do nothing.

Related links:

* :c:func:`ncclGroupStart`
* :c:func:`ncclGroupEnd`

Group Operation Ordering Semantics
-------------------------------------

Although NCCL group allows different operations to be issued in one shot, users still need to guarantee the same
issuing order of the operations among different GPUs no matter whether the operations are issued to the same or
different communicators.

For example, the following code provides the correct order of the operations. In this example, *comm0* and *comm1*
are duplicated independent communicators that include rank 0 and 1.

.. code:: C

 RANK0/GPU0/Process0:
 ncclGroupStart();
 ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
 ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, ncclSum, comm0, stream);
 ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, ncclSum, comm0, stream);
 ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, ncclSum, comm1, stream);
 ncclGroupEnd();

 RANK1/GPU1/Process1:
 ncclGroupStart();
 ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
 ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, ncclSum, comm0, stream);
 ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, ncclSum, comm0, stream);
 ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, ncclSum, comm1, stream);
 ncclGroupEnd();

However, changing the order of any operations will lead to incorrect results or hang as shown in the following 2 examples:

.. code:: C

 RANK0/GPU0/Process0:
 ncclGroupStart();
 ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
 ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, ncclSum, comm0, stream); // WRONG: reversed order
 ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, ncclSum, comm0, stream); // WRONG: reversed order
 ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, ncclSum, comm1, stream);
 ncclGroupEnd();

 RANK1/GPU1/Process1:
 ncclGroupStart();
 ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
 ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, ncclSum, comm0, stream); // WRONG: reversed order
 ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, ncclSum, comm0, stream); // WRONG: reversed order
 ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, ncclSum, comm1, stream);
 ncclGroupEnd();

.. code:: C

 RANK0/GPU0/Process0:
 ncclGroupStart();
 ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, ncclSum, comm1, stream); // WRONG: reversed order
 ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
 ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, ncclSum, comm0, stream);
 ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, ncclSum, comm0, stream);
 ncclGroupEnd();

 RANK1/GPU1/Process1:
 ncclGroupStart();
 ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
 ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, ncclSum, comm0, stream);
 ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, ncclSum, comm0, stream);
 ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, ncclSum, comm1, stream); // WRONG: reversed order
 ncclGroupEnd();

Nonblocking Group Operation
-------------------------------------

If a communicator is marked as nonblocking through :c:func:`ncclCommInitRankConfig`, the group functions become asynchronous
correspondingly. In this case, if users issue multiple NCCL operations in one group, returning from :c:func:`ncclGroupEnd` might
not mean the NCCL communication kernels have been issued to CUDA streams. If :c:func:`ncclGroupEnd` returns :c:enumerator:`ncclSuccess <ncclResult_t.ncclSuccess>`, it means
NCCL kernels have been issued to streams; if it returns :c:enumerator:`ncclInProgress <ncclResult_t.ncclInProgress>`, it means NCCL kernels are being issued to streams
in the background. It is users' responsibility to make sure the state of the communicator changes into :c:enumerator:`ncclSuccess <ncclResult_t.ncclSuccess>`
before calling related CUDA calls (e.g. ``cudaStreamSynchronize``):

.. code:: C

 ncclGroupStart();
   for (int g=0; g<ngpus; g++) {
     ncclAllReduce(sendbuffs[g]+offsets[i], recvbuffs[g]+offsets[i], counts[i], datatype[i], ncclSum, comms[g], streams[g]);
   }
 ret = ncclGroupEnd();
 if (ret == ncclInProgress) {
    for (int g=0; g<ngpus; g++) {
      do {
        ncclCommGetAsyncError(comms[g], &state);
      } while (state == ncclInProgress);
    }
 } else if (ret == ncclSuccess) {
    /* Successfully issued */
    printf("NCCL kernel issue succeeded\n");
 } else {
    /* Errors happen */
    reportErrorAndRestart();
 }

 for (int g=0; g<ngpus; g++) {
   cudaStreamSynchronize(streams[g]);
 }

Related links:

* :c:func:`ncclCommInitRankConfig`
* :c:func:`ncclCommGetAsyncError`
