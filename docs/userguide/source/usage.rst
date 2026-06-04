##########
Using NCCL
##########

Using NCCL is similar to using any other library in your code:

1. Install the NCCL library on your system

2. Modify your application to link to that library

3. Include the header file nccl.h in your application

4. Create a communicator (see :ref:`communicator-label`)

5. Use NCCL collective communication primitives to perform data communication. You can familiarize yourself with the :ref:`api-label` documentation to maximize your usage performance.

Collective communication primitives are common patterns of data transfer among a group of CUDA devices. A communication algorithm involves many processors that are communicating together.
Each CUDA device is identified within the communication group by a zero-based index or rank. Each rank uses a communicator object to refer to the collection of GPUs that are intended to work together.
The creation of a communicator is the first step needed before launching any communication operation.

.. toctree::
   :maxdepth: 2

   usage/communicators
   usage/collectives
   usage/data
   usage/streams
   usage/groups
   usage/p2p
   usage/threadsafety
   usage/inplace
   usage/cudagraph
   usage/bufferreg
   usage/deviceapi
