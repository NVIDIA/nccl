########
Examples
########

The examples in this section provide an overall view of how to use NCCL in various environments, combining one or multiple techniques:


* using multiple GPUs per thread/process
* using multiple threads
* using multiple processes - the examples with multiple processes use MPI as parallel runtime environment, but any multi-process system should be able to work similarly.


Ensure that you always check the return codes from the NCCL functions.  For clarity, the following examples do not contain error checking.

**********************************************
Communicator Creation and Destruction Examples
**********************************************

The following examples demonstrate common use cases for NCCL initialization.


Example 1: Single Process, Single Thread, Multiple Devices
----------------------------------------------------------


In the specific case of a single process, ncclCommInitAll can be used. Here is an example creating a communicator for 4 devices, therefore, there are 4 communicator objects:

.. code:: C

 ncclComm_t comms[4];
 int devs[4] = { 0, 1, 2, 3 };
 ncclCommInitAll(comms, 4, devs);

Next, you can call NCCL collective operations using a single thread and group calls, or multiple threads, each provided with a comm object.


At the end of the program, all of the communicator objects are destroyed:

.. code:: C

 for (int i=0; i<4; i++)
   ncclCommDestroy(comms[i]);

The following code depicts a complete working example with a single process that manages multiple devices:

.. code:: C

 #include <stdlib.h>
 #include <stdio.h>
 #include "cuda_runtime.h"
 #include "nccl.h"

 #define CUDACHECK(cmd) do {                         \
   cudaError_t err = cmd;                            \
   if (err != cudaSuccess) {                         \
     printf("Failed: Cuda error %s:%d '%s'\n",       \
         __FILE__,__LINE__,cudaGetErrorString(err)); \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)


 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t res = cmd;                           \
   if (res != ncclSuccess) {                         \
     printf("Failed, NCCL error %s:%d '%s'\n",       \
         __FILE__,__LINE__,ncclGetErrorString(res)); \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)


 int main(int argc, char* argv[])
 {
   ncclComm_t comms[4];


   //managing 4 devices
   int nDev = 4;
   int size = 32*1024*1024;
   int devs[4] = { 0, 1, 2, 3 };


   //allocating and initializing device buffers
   float** sendbuff = (float**)malloc(nDev * sizeof(float*));
   float** recvbuff = (float**)malloc(nDev * sizeof(float*));
   cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


   for (int i = 0; i < nDev; ++i) {
     CUDACHECK(cudaSetDevice(i));
     CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
     CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
     CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
     CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
     CUDACHECK(cudaStreamCreate(s+i));
   }


   //initializing NCCL
   NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
   NCCLCHECK(ncclGroupStart());
   for (int i = 0; i < nDev; ++i)
     NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
         comms[i], s[i]));
   NCCLCHECK(ncclGroupEnd());


   //synchronizing on CUDA streams to wait for completion of NCCL operation
   for (int i = 0; i < nDev; ++i) {
     CUDACHECK(cudaSetDevice(i));
     CUDACHECK(cudaStreamSynchronize(s[i]));
   }


   //free device buffers
   for (int i = 0; i < nDev; ++i) {
     CUDACHECK(cudaSetDevice(i));
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
   }


   //finalizing NCCL
   for(int i = 0; i < nDev; ++i)
       ncclCommDestroy(comms[i]);


   printf("Success \n");
   return 0;
 }

Example 2: One Device per Process or Thread
-------------------------------------------

When a process or host thread is responsible for at most one GPU, ncclCommInitRank can be used as a collective call to create a communicator. Each thread or process will get its own object.


The following code is an example of a communicator creation in the context of MPI, using one device per MPI rank.


First, we retrieve MPI information about processes:

.. code:: C

 int myRank, nRanks;
 MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
 MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

Next, a single rank will create a unique ID and send it to all other ranks to make sure everyone has it:

.. code:: C

 ncclUniqueId id;
 if (myRank == 0) ncclGetUniqueId(&id);
 MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

Finally, we create the communicator:

.. code:: C

 ncclComm_t comm;
 ncclCommInitRank(&comm, nRanks, id, myRank);

We can now call the NCCL collective operations using the communicator.

.. code:: C

 ncclAllReduce( ... , comm);

Finally, we destroy the communicator object:

.. code:: C

 ncclCommDestroy(comm);


The following code depicts a complete working example with multiple MPI processes and one device per process:

.. code:: C

 #include <stdio.h>
 #include "cuda_runtime.h"
 #include "nccl.h"
 #include "mpi.h"
 #include <unistd.h>
 #include <stdint.h>
 #include <stdlib.h>


 #define MPICHECK(cmd) do {                          \
   int e = cmd;                                      \
   if( e != MPI_SUCCESS ) {                          \
     printf("Failed: MPI error %s:%d '%d'\n",        \
         __FILE__,__LINE__, e);   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)


 #define CUDACHECK(cmd) do {                         \
   cudaError_t e = cmd;                              \
   if( e != cudaSuccess ) {                          \
     printf("Failed: Cuda error %s:%d '%s'\n",             \
         __FILE__,__LINE__,cudaGetErrorString(e));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)


 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t r = cmd;                             \
   if (r!= ncclSuccess) {                            \
     printf("Failed, NCCL error %s:%d '%s'\n",             \
         __FILE__,__LINE__,ncclGetErrorString(r));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)


 static uint64_t getHash(const char* string, size_t n) {
   // Based on DJB2a, result = result * 33 ^ char
   uint64_t result = 5381;
   for (size_t c = 0; c < n; c++){
     result = ((result << 5) + result) ^ string[c];
   }
   return result;
 }

 /* Generate a hash of the unique identifying string for this host
  * that will be unique for both bare-metal and container instances
  * Equivalent of a hash of;
  *
  * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
  *
  */
 #define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
 static uint64_t getHostHash(const char* hostname) {
   char hostHash[1024];

   // Fall back is the hostname if something fails
   (void) strncpy(hostHash, hostname, sizeof(hostHash));
   int offset = strlen(hostHash);

   FILE *file = fopen(HOSTID_FILE, "r");
   if (file != NULL) {
     char *p;
     if (fscanf(file, "%ms", &p) == 1) {
	 strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
	 free(p);
     }
   }
   fclose(file);

   // Make sure the string is terminated
   hostHash[sizeof(hostHash)-1]='\0';

   return getHash(hostHash, strlen(hostHash));
 }

 static void getHostName(char* hostname, int maxlen) {
   gethostname(hostname, maxlen);
   for (int i=0; i< maxlen; i++) {
     if (hostname[i] == '.') {
         hostname[i] = '\0';
         return;
     }
   }
 }


 int main(int argc, char* argv[])
 {
   int size = 32*1024*1024;


   int myRank, nRanks, localRank = 0;


   //initializing MPI
   MPICHECK(MPI_Init(&argc, &argv));
   MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
   MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


   //calculating localRank based on hostname which is used in selecting a GPU
   uint64_t hostHashs[nRanks];
   char hostname[1024];
   getHostName(hostname, 1024);
   hostHashs[myRank] = getHostHash(hostname);
   MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
   for (int p=0; p<nRanks; p++) {
      if (p == myRank) break;
      if (hostHashs[p] == hostHashs[myRank]) localRank++;
   }


   ncclUniqueId id;
   ncclComm_t comm;
   float *sendbuff, *recvbuff;
   cudaStream_t s;


   //get NCCL unique ID at rank 0 and broadcast it to all others
   if (myRank == 0) ncclGetUniqueId(&id);
   MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


   //picking a GPU based on localRank, allocate device buffers
   CUDACHECK(cudaSetDevice(localRank));
   CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
   CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
   CUDACHECK(cudaStreamCreate(&s));


   //initializing NCCL
   NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


   //communicating using NCCL
   NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
         comm, s));


   //completing NCCL operation by synchronizing on the CUDA stream
   CUDACHECK(cudaStreamSynchronize(s));


   //free device buffers
   CUDACHECK(cudaFree(sendbuff));
   CUDACHECK(cudaFree(recvbuff));


   //finalizing NCCL
   ncclCommDestroy(comm);


   //finalizing MPI
   MPICHECK(MPI_Finalize());


   printf("[MPI Rank %d] Success \n", myRank);
   return 0;
 }

.. _Ex3:

Example 3: Multiple Devices per Thread
--------------------------------------

You can combine both multiple process or threads and multiple device per process or thread. In this case, we need to use group semantics.


The following example combines MPI and multiple devices per process (=MPI rank).


First, we retrieve MPI information about processes:

.. code:: C

 int myRank, nRanks;
 MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
 MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

Next, a single rank will create a unique ID and send it to all other ranks to make sure everyone has it:

.. code:: C

 ncclUniqueId id;
 if (myRank == 0) ncclGetUniqueId(&id);
 MPI_Bcast(id, sizeof(id), MPI_BYTE, 0, 0, MPI_COMM_WORLD);

Then, we create our ngpus communicator objects, which are part of a larger group of ngpus*nRanks:

.. code:: C

 ncclComm_t comms[ngpus];
 ncclGroupStart();
 for (int i=0; i<ngpus; i++) {
   cudaSetDevice(devs[i]);
   ncclCommInitRank(comms+i, ngpus*nRanks, id, myRank*ngpus+i);
 }
 ncclGroupEnd();

Next, we call NCCL collective operations using a single thread and group calls, or multiple threads, each provided with a comm object.

At the end of the program, we destroy all communicators objects:

.. code:: C

 for (int i=0; i<ngpus; i++)
   ncclCommDestroy(comms[i]);

The following code depicts a complete working example with multiple MPI processes and multiple devices per process:

.. code:: C

 #include <stdio.h>
 #include "cuda_runtime.h"
 #include "nccl.h"
 #include "mpi.h"
 #include <unistd.h>
 #include <stdint.h>


 #define MPICHECK(cmd) do {                          \
   int e = cmd;                                      \
   if( e != MPI_SUCCESS ) {                          \
     printf("Failed: MPI error %s:%d '%d'\n",        \
         __FILE__,__LINE__, e);   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)


 #define CUDACHECK(cmd) do {                         \
   cudaError_t e = cmd;                              \
   if( e != cudaSuccess ) {                          \
     printf("Failed: Cuda error %s:%d '%s'\n",             \
         __FILE__,__LINE__,cudaGetErrorString(e));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)


 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t r = cmd;                             \
   if (r!= ncclSuccess) {                            \
     printf("Failed, NCCL error %s:%d '%s'\n",             \
         __FILE__,__LINE__,ncclGetErrorString(r));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)


 static uint64_t getHash(const char* string) {
   // Based on DJB2a, result = result * 33 ^ char
   uint64_t result = 5381;
   for (int c = 0; string[c] != '\0'; c++){
     result = ((result << 5) + result) ^ string[c];
   }
   return result;
 }

 /* Generate a hash of the unique identifying string for this host
  * that will be unique for both bare-metal and container instances
  * Equivalent of a hash of;
  *
  * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
  *
  */
 #define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
 static uint64_t getHostHash(const char* hostname) {
   char hostHash[1024];

   // Fall back is the hostname if something fails
   (void) strncpy(hostHash, hostname, sizeof(hostHash));
   int offset = strlen(hostHash);

   FILE *file = fopen(HOSTID_FILE, "r");
   if (file != NULL) {
     char *p;
     if (fscanf(file, "%ms", &p) == 1) {
	 strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
	 free(p);
     }
   }
   fclose(file);

   // Make sure the string is terminated
   hostHash[sizeof(hostHash)-1]='\0';

   return getHash(hostHash, strlen(hostHash));
 }

 static void getHostName(char* hostname, int maxlen) {
   gethostname(hostname, maxlen);
   for (int i=0; i< maxlen; i++) {
     if (hostname[i] == '.') {
         hostname[i] = '\0';
         return;
     }
   }
 }


 int main(int argc, char* argv[])
 {
   int size = 32*1024*1024;


   int myRank, nRanks, localRank = 0;


   //initializing MPI
   MPICHECK(MPI_Init(&argc, &argv));
   MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
   MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


   //calculating localRank which is used in selecting a GPU
   uint64_t hostHashs[nRanks];
   char hostname[1024];
   getHostName(hostname, 1024);
   hostHashs[myRank] = getHostHash(hostname);
   MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
   for (int p=0; p<nRanks; p++) {
      if (p == myRank) break;
      if (hostHashs[p] == hostHashs[myRank]) localRank++;
   }


   //each process is using two GPUs
   int nDev = 2;


   float** sendbuff = (float**)malloc(nDev * sizeof(float*));
   float** recvbuff = (float**)malloc(nDev * sizeof(float*));
   cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


   //picking GPUs based on localRank
   for (int i = 0; i < nDev; ++i) {
     CUDACHECK(cudaSetDevice(localRank*nDev + i));
     CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
     CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
     CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
     CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
     CUDACHECK(cudaStreamCreate(s+i));
   }


   ncclUniqueId id;
   ncclComm_t comms[nDev];


   //generating NCCL unique ID at one process and broadcasting it to all
   if (myRank == 0) ncclGetUniqueId(&id);
   MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


   //initializing NCCL, group API is required around ncclCommInitRank as it is
   //called across multiple GPUs in each thread/process
   NCCLCHECK(ncclGroupStart());
   for (int i=0; i<nDev; i++) {
      CUDACHECK(cudaSetDevice(localRank*nDev + i));
      NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));
   }
   NCCLCHECK(ncclGroupEnd());


   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread/process
   NCCLCHECK(ncclGroupStart());
   for (int i=0; i<nDev; i++)
      NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
            comms[i], s[i]));
   NCCLCHECK(ncclGroupEnd());


   //synchronizing on CUDA stream to complete NCCL communication
   for (int i=0; i<nDev; i++)
       CUDACHECK(cudaStreamSynchronize(s[i]));


   //freeing device memory
   for (int i=0; i<nDev; i++) {
      CUDACHECK(cudaFree(sendbuff[i]));
      CUDACHECK(cudaFree(recvbuff[i]));
   }


   //finalizing NCCL
   for (int i=0; i<nDev; i++) {
      ncclCommDestroy(comms[i]);
   }


   //finalizing MPI
   MPICHECK(MPI_Finalize());


   printf("[MPI Rank %d] Success \n", myRank);
   return 0;
 }

.. _Ex4:

Example 4: Multiple communicators per device
--------------------------------------------

NCCL allows users to create multiple communicators per device. The following code shows an example with multiple MPI processes, one device per process, and multiple communicators per device:

.. code:: C

  // blocking communicators
  CUDACHECK(cudaSetDevice(localRank));
  for (int i = 0; i < commNum; ++i) {
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRank(&blockingComms[i], nRanks, id, myRank));
  }

  // non-blocking communicators
  CUDACHECK(cudaSetDevice(localRank));
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  for (int i = 0; i < commNum; ++i) {
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRankConfig(&nonblockingComms[i], nRanks, id, myRank, &config));
    do {
      NCCLCHECK(ncclCommGetAsyncError(nonblockingComms[i], &state));
    } while(state == ncclInProgress && checkTimeout() != true);
  }

`checkTimeout()` should be a user-defined function. For more nonblocking communicator usage, please check :ref:`ft`.
In addition, if you want to split communicators instead of creating a new one, please check :c:func:`ncclCommSplit`.

**********************
Communication Examples
**********************

The following examples demonstrate common patterns for executing NCCL collectives.


Example 1: One Device per Process or Thread
-------------------------------------------


If you have a thread or process per device, then each thread calls the collective operation for its device, for example, AllReduce:

.. code:: C

 ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);


After the call, the operation has been enqueued to the stream.  Therefore, you can call cudaStreamSynchronize if you want to wait for the operation to be complete:

.. code:: C

 cudaStreamSynchronize(stream);

For a complete working example with MPI and single device per MPI process, see “Example 2: One Device per Process or Thread”.

Example 2: Multiple Devices per Thread
--------------------------------------

When a single thread manages multiple devices, you need to use group semantics to launch the operation on multiple devices at once:

.. code:: C

 ncclGroupStart();
 for (int i=0; i<ngpus; i++)
   ncclAllReduce(sendbuffs[i], recvbuff[i], count, datatype, op, comms[i], streams[i]);
 ncclGroupEnd();

After ncclGroupEnd, all of the operations have been enqueued to the stream.  Therefore, you can now call cudaStreamSynchronize if you want to wait for the operation to be complete:


.. code:: C

 for (int i=0; i<ngpus; i++)
   cudaStreamSynchronize(streams[i]);

For a complete working example with MPI and multiple devices per MPI process, see :ref:`Example 3: Multiple Devices per Thread<Ex3>`.
