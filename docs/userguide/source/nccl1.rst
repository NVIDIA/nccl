###############################
Migrating from NCCL 1 to NCCL 2
###############################

If you are using NCCL 1.x and want to move to NCCL 2.x, be aware that the APIs have changed slightly. NCCL 2.x supports
all of the collectives that NCCL 1.x supports, but with slight modifications to the API.

In addition, NCCL 2.x also requires the usage of the “Group API” when a single thread manages NCCL calls for multiple
GPUs.

The following list summarizes the changes that may be required in usage of NCCL API when using an application that has a
single thread that manages NCCL calls for multiple GPUs, and is ported from NCCL 1.x to 2.x:

Initialization
--------------

In versions 1.x, NCCL had to be initialized using ncclCommInitAll at a single thread or having one thread per GPU
concurrently call ncclCommInitRank. NCCL 2.x retains these two modes of initialization. It adds a new mode with the
Group API where ncclCommInitRank can be called in a loop, like a communication call, as shown below. The loop has to be
guarded by the Group start and end API.

.. code:: C

 ncclGroupStart();
 for (int i=0; i<ngpus; i++) {
   cudaSetDevice(i);
   ncclCommInitRank(comms+i, ngpus, id, i);
 }
 ncclGroupEnd();


Communication
-------------

In NCCL 2.x, the collective operation can be initiated for different devices by making calls in a loop, on a single
thread. This is similar to the usage in NCCL 1.x. However, this loop has to be guarded by the Group API in 2.x. Unlike
in 1.x, the application does not have to select the relevant CUDA device before making the communication API call. NCCL
runtime internally selects the device associated with the NCCL communicator handle. For example:

.. code:: C

 ncclGroupStart();
 for (int i=0; i<nLocalDevs; i++) {
   ncclAllReduce(..., comm[i], stream[i]);
 }
 ncclGroupEnd();

When using only one device per thread or one device per process, the general usage of the API remains unchanged from NCCL
1.x to 2.x. The usage of the group API is not required in this case.

Counts
------
Counts provided as arguments are now of type size_t instead of integer.

In-place usage for AllGather and ReduceScatter
----------------------------------------------
For more information, see “In-place Operations”.

AllGather arguments order
-------------------------
The AllGather function had its arguments reordered. The prototype changed from:

.. code:: C

 ncclResult_t  ncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream);

to:

.. code:: C

 ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

The recvbuff argument has been moved after sendbuff to be consistent with all the other operations.

Datatypes
---------

New datatypes have been added in NCCL 2.x.  The ones present in NCCL 1.x did not change and are still usable in NCCL 2.x.

Error codes
-----------

Error codes have been merged into the ncclInvalidArgument category and have been simplified. A new ncclInvalidUsage code has been created to cover new programming errors.
