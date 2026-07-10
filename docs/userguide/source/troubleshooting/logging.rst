*******
Logging
*******

NCCL provides configurable logging to help diagnose issues, understand runtime behavior and gain insight into the choices NCCL makes during execution (such as algorithm selection, topology detection, and network configuration).

Logging Environment Variables
=============================

The following environment variables control NCCL logging behavior:

+------------------------------------+-----------------------------------------------+
| Variable                           | Description                                   |
+====================================+===============================================+
| :ref:`NCCL_DEBUG`                  | Sets verbosity: VERSION, WARN, INFO, TRACE    |
+------------------------------------+-----------------------------------------------+
| :ref:`NCCL_DEBUG_SUBSYS`           | Filters log output by subsystem               |
+------------------------------------+-----------------------------------------------+
| :ref:`NCCL_DEBUG_FILE`             | Writes logs to files (%h host, %p PID)        |
+------------------------------------+-----------------------------------------------+
| :ref:`NCCL_DEBUG_TIMESTAMP_FORMAT` | Sets timestamp format (strftime syntax)       |
+------------------------------------+-----------------------------------------------+
| :ref:`NCCL_DEBUG_TIMESTAMP_LEVELS` | Selects which levels include timestamps       |
+------------------------------------+-----------------------------------------------+

.. highlight:: shell

Logging Levels
==============

NCCL supports several logging levels, from least to most verbose:

+---------+--------------------------------------------+-----------------------------+
| Level   | Description                                | Use Case                    |
+=========+============================================+=============================+
| VERSION | Prints NCCL version at startup             | Verify installation         |
+---------+--------------------------------------------+-----------------------------+
| WARN    | Warnings and errors                        | Production minimum          |
+---------+--------------------------------------------+-----------------------------+
| INFO    | Detailed operational information           | Diagnose runtime issues     |
+---------+--------------------------------------------+-----------------------------+
| TRACE   | Replayable traces, plus CALL APIs          | Deep debugging / NCCL dev   |
+---------+--------------------------------------------+-----------------------------+

Setting the Logging Level
-------------------------

Set the ``NCCL_DEBUG`` environment variable:

.. code:: shell

    NCCL_DEBUG=INFO ./my_app

Example Output
--------------

The snippets below are excerpts from NCCL logs; only the relevant lines are shown. Exact values, line numbers, and formatting can vary by NCCL version and environment.

**NCCL_DEBUG=VERSION** - Prints NCCL version at startup:

.. code:: shell

    NCCL version 2.30.3+cuda13.0

**NCCL_DEBUG=WARN** - Warnings and errors are printed:

.. code:: shell

    [2026-05-05 06:16:47] node-01:189884:189884 [3] plugin/net.cc:334 NCCL WARN Failed to initialize any NET plugin

**NCCL_DEBUG=INFO** - Detailed information about NCCL operations:

.. code:: shell

    node-01:3873285:3873285 [0] NCCL INFO Initialized NET plugin IB
    node-01:3873285:3873285 [0] NCCL INFO Assigned NET plugin IB to comm
    node-01:3873285:3873285 [0] NCCL INFO Assigned GIN plugin GIN_XXXX to comm
    node-01:3873285:3873285 [0] NCCL INFO Assigned RMA plugin RMA_XXXX to comm
    node-01:3873285:3873285 [0] NCCL INFO Using network IB
    node-01:3873285:3873285 [0] NCCL INFO DMA-BUF is available on GPU device 0

**NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=CALL** - Function call tracing:

.. code:: shell

    node-01:986207:986207 NCCL CALL ncclGroupStart()
    node-01:986207:986207 NCCL CALL ncclSend(0,0x...,8388608,7,0,2,0x...,0x...)
    node-01:986207:986207 NCCL CALL ncclRecv(0,0x...,8388608,7,0,0,0x...,0x...)
    node-01:986207:986207 NCCL CALL ncclGroupEnd()

Filtering by Subsystem
======================

When using ``NCCL_DEBUG=INFO`` or ``NCCL_DEBUG=TRACE``, output can be filtered to include specific
subsystems using ``NCCL_DEBUG_SUBSYS``. This helps focus on relevant information without
being overwhelmed by unrelated messages.

Basic Usage
-----------

.. code:: shell

    # Only show network-related messages
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET ./my_app

    # Show multiple subsystems
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH ./my_app

    # Show everything except verbose proxy messages
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=^PROXY ./my_app

    # Exclude multiple subsystems
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=^PROXY,ALLOC ./my_app

Available Subsystems
--------------------

+-----------+------------------------------------------------------------------------+
| Subsystem | Description                                                            |
+===========+========================================================================+
| INIT      | Initialization and setup                                               |
+-----------+------------------------------------------------------------------------+
| COLL      | Collective operations (AllReduce, Broadcast, etc.)                     |
+-----------+------------------------------------------------------------------------+
| P2P       | Peer-to-peer send/receive operations                                   |
+-----------+------------------------------------------------------------------------+
| SHM       | Shared memory transport                                                |
+-----------+------------------------------------------------------------------------+
| NET       | Network transport (IB, sockets, etc.)                                  |
+-----------+------------------------------------------------------------------------+
| GRAPH     | Topology detection and graph search                                    |
+-----------+------------------------------------------------------------------------+
| TUNING    | Algorithm and protocol selection                                       |
+-----------+------------------------------------------------------------------------+
| ENV       | Environment variable processing                                        |
+-----------+------------------------------------------------------------------------+
| ALLOC     | Memory allocations                                                     |
+-----------+------------------------------------------------------------------------+
| CALL      | Function call tracing                                                  |
+-----------+------------------------------------------------------------------------+
| PROXY     | Proxy thread operations                                                |
+-----------+------------------------------------------------------------------------+
| NVLS      | NVLink SHARP operations                                                |
+-----------+------------------------------------------------------------------------+
| BOOTSTRAP | Early initialization and bootstrapping                                 |
+-----------+------------------------------------------------------------------------+
| REG       | Buffer registration                                                    |
+-----------+------------------------------------------------------------------------+
| PROFILE   | Coarse-grained initialization profiling                                |
+-----------+------------------------------------------------------------------------+
| RAS       | Reliability, availability, and serviceability                          |
+-----------+------------------------------------------------------------------------+
| DESTROY   | Communicator destroy, abort, revoke, and plugin unload/close           |
|           | operations                                                             |
+-----------+------------------------------------------------------------------------+
| ALL       | All subsystems                                                         |
+-----------+------------------------------------------------------------------------+

The default subsystems (when ``NCCL_DEBUG_SUBSYS`` is not set) are ``INIT,BOOTSTRAP,ENV``.

Example Output by Subsystem
----------------------------

The following examples show typical output for each subsystem (output is truncated for brevity):

**INIT** - Shows initialization and plugin assignment:

.. code:: shell

    node-01:1895902:1895913 [0] NCCL INFO Initialized NET plugin IB
    node-01:1895902:1895913 [0] NCCL INFO Assigned NET plugin IB to comm
    node-01:1895902:1895913 [0] NCCL INFO Assigned GIN plugin GIN_XXXX to comm
    node-01:1895902:1895913 [0] NCCL INFO Assigned RMA plugin RMA_XXXX to comm
    node-01:1895902:1895913 [0] NCCL INFO Using network IB
    node-01:1895902:1895913 [0] NCCL INFO DMA-BUF is available on GPU device 0
    node-01:1895902:1895913 [0] NCCL INFO [Rank 0] ncclCommInitRankConfig comm 0x... rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId XXXX commId 0x... - Init START
    node-01:1895902:1895913 [0] NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 0 is X,X,X,X,X,X,X,X. (GPU affinity = X,X,X,X,X,X,X,X ; CPU affinity = X-X).

**GRAPH** - Shows topology detection and communication patterns:

.. code:: shell

    node-01:1895811:1895822 [0] NCCL INFO Tree 0 : -1 -> 0 -> 1/-1/-1
    node-01:1895811:1895822 [0] NCCL INFO Tree 1 : -1 -> 0 -> 1/-1/-1
    node-01:1895811:1895822 [0] NCCL INFO Ring 00 : 1 -> 0 -> 1
    node-01:1895811:1895822 [0] NCCL INFO Ring 01 : 1 -> 0 -> 1

**COLL** - Shows collective operation details:

.. code:: shell

    node-01:1896279:1896279 [0] NCCL INFO AllReduce: opCount 0 sendbuff 0x... recvbuff 0x... count 2 datatype 7 op 0 root 0 comm 0x... [nranks=2] stream 0x...
    node-01:1896279:1896279 [0] NCCL INFO AllReduce: opCount 0 sendbuff 0x... recvbuff 0x... count 2 datatype 7 op 0 root 0 comm 0x... [nranks=2] stream 0x...

**SHM** - Shows shared memory transport operations:

.. code:: shell

    node-01:1896458:1896477 [0] NCCL INFO MMAP allocated shareable host buffer /dev/shm/nccl-XXXXXX size 4096 ptr 0x...
    node-01:1896458:1896480 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
    node-01:1896458:1896477 [0] NCCL INFO MMAP allocated shareable host buffer /dev/shm/nccl-XXXXXX size 4096 ptr 0x...
    node-01:1896458:1896480 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct

**RAS** - Shows reliability, availability, and serviceability information:

.. code:: shell

    node-01:1896503:1896518 [0] NCCL INFO RAS thread started
    node-01:1896503:1896518 [0] NCCL INFO RAS handling local addRanks request (old nRasPeers 0)
    node-01:1896503:1896518 [0] NCCL INFO RAS finished local processing of addRanks request (new nRasPeers 1, nRankPeers 1)

**DESTROY** - Shows communicator teardown and plugin unload/close operations:

.. code:: shell

    node-01:2056033:2056033 [1] NCCL INFO comm 0x... rank 1 nranks 4 cudaDev 1 busId XXXX - Destroy COMPLETE
    node-01:2056035:2056035 [3] NCCL INFO comm 0x... rank 3 nranks 4 cudaDev 3 busId XXXX - Destroy COMPLETE
    node-01:2056032:2056032 [0] NCCL INFO comm 0x... rank 0 nranks 4 cudaDev 0 busId XXXX - Destroy COMPLETE
    node-01:2056034:2056034 [2] NCCL INFO comm 0x... rank 2 nranks 4 cudaDev 2 busId XXXX - Destroy COMPLETE
    node-01:2056035:2056035 [3] NCCL INFO ENV/Plugin: Closing env plugin ncclEnvDefault
    node-01:2056033:2056033 [1] NCCL INFO ENV/Plugin: Closing env plugin ncclEnvDefault
    node-01:2056034:2056034 [2] NCCL INFO ENV/Plugin: Closing env plugin ncclEnvDefault
    node-01:2056032:2056032 [0] NCCL INFO ENV/Plugin: Closing env plugin ncclEnvDefault

**PROXY** - Shows proxy thread operations:

.. code:: shell

    node-01:1896541:1896559 [0] NCCL INFO New proxy recv connection 0 from local rank 0, transport 1
    node-01:1896541:1896559 [0] NCCL INFO proxyProgressAsync opId=0x... op.type=1 op.reqBuff=0x... op.respSize=16 done
    node-01:1896541:1896562 [0] NCCL INFO ncclPollProxyResponse Received new opId=0x...
    node-01:1896541:1896562 [0] NCCL INFO resp.opId=0x... matches expected opId=0x...
    node-01:1896541:1896562 [0] NCCL INFO Connected to proxy localRank 0 -> connection 0x...
    node-01:1896541:1896559 [0] NCCL INFO Received and initiated operation=Init res=0

**ENV** - Shows environment variable processing:

.. code:: shell

    node-01:1896826:1896838 [0] NCCL INFO NCCL_SHM_DISABLE set by environment to 1.

**REG** - Shows buffer registration:

.. code:: shell

    node-01:1896963:1896984 [0] NCCL INFO register comm 0x... buffer 0x... size 8
    node-01:1896963:1896984 [0] NCCL INFO register comm 0x... buffer 0x... size 8

**ALLOC** - Shows memory allocations:

.. code:: shell

    node-01:1120933:1120933 [0] NCCL INFO init.cc:2353 Cuda Host Alloc Size 4 pointer 0x...
    node-01:1120933:1120933 [0] NCCL INFO MemManager: Initialized for device 0
    node-01:1120933:1120933 [0] NCCL INFO misc/utils.cc:297 memory stack hunk malloc(65536)
    node-01:1120933:1120933 [0] NCCL INFO Mem Realloc old size 0, new size 256 pointer 0x...
    node-01:1120933:1120950 [0] NCCL INFO Mem Realloc old size 0, new size 32 pointer 0x...
    node-01:1120933:1120933 [0] NCCL INFO channel.cc:43 Cuda CallocAsync Size 608 pointer 0x... memType 2
    node-01:1120933:1120933 [0] NCCL INFO channel.cc:46 Cuda CallocAsync Size 24 pointer 0x... memType 2
    node-01:1120933:1120933 [0] NCCL INFO channel.cc:58 Cuda CallocAsync Size 4 pointer 0x... memType 2
    node-01:1120933:1120933 [0] NCCL INFO channel.cc:43 Cuda CallocAsync Size 608 pointer 0x... memType 2
    node-01:1120933:1120933 [0] NCCL INFO channel.cc:46 Cuda CallocAsync Size 24 pointer 0x... memType 2
    node-01:1120933:1120933 [0] NCCL INFO channel.cc:58 Cuda CallocAsync Size 4 pointer 0x... memType 2

**CALL** - Shows function call tracing (requires ``NCCL_DEBUG=TRACE``):

.. code:: shell

    node-01:1897248:1897248 NCCL CALL ncclGroupStart()
    node-01:1897248:1897248 NCCL CALL ncclAllReduce(0x...,0x...,2,7,0,0,0x...,0x...)
    node-01:1897248:1897248 NCCL CALL ncclAllReduce(0x...,0x...,2,7,0,0,0x...,0x...)
    node-01:1897248:1897248 NCCL CALL ncclGroupEnd()

**BOOTSTRAP** - Shows early initialization and bootstrapping:

.. code:: shell

    node-01:1897326:1897337 [0] NCCL INFO Bootstrap timings total 0.001447 (create 0.000081, send 0.000255, recv 0.000268, ring 0.000068, delay 0.000000)

**PROFILE** - Shows coarse-grained initialization profiling:

.. code:: shell

    node-01:1897505:1897517 [0] NCCL INFO Bootstrap timings total 0.001204 (create 0.000080, send 0.000303, recv 0.000137, ring 0.000063, delay 0.000000)
    node-01:1897505:1897517 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 2 total 0.37 (kernels 0.28, alloc 0.05, bootstrap 0.00, allgathers 0.00, topo 0.03, graphs 0.00, connections 0.01, rest 0.00)

**NET** - Shows network transport operations:

.. code:: shell

    node-01:1897709:1897731 [0] NCCL INFO Connected to proxy localRank 0 -> connection 0x...
    node-01:1897709:1897731 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] [send] via NET/Socket/0
    node-01:1897709:1897727 [0] NCCL INFO New proxy send connection 3 from local rank 0, transport 2
    node-01:1897709:1897731 [0] NCCL INFO Connected to proxy localRank 0 -> connection 0x...
    node-01:1897709:1897731 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] [send] via NET/Socket/0

**P2P** - Shows peer-to-peer operations:

.. code:: shell

    node-01:631398:631465 [0] NCCL INFO Allocated shareable buffer 0x... size 10485760 ipcDesc 0x...
    node-01:631398:631398 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
    node-01:631398:631465 [0] NCCL INFO Allocated shareable buffer 0x... size 2097152 ipcDesc 0x...
    node-01:631398:631398 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM

**TUNING** - Shows algorithm and protocol selection:

.. code:: shell

    node-01:631871:631871 [0] NCCL INFO AllReduce: 33554432 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..31}
    node-01:631871:631871 [0] NCCL INFO AllReduce: 33554432 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..31}
    node-01:631871:631871 [0] NCCL INFO AllReduce: 33554432 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..31}

**NVLS** - Shows NVLink SHARP operations:

.. code:: shell

    node-01:1538174:1538174 [0] NCCL INFO NVLS Creating Multicast group nranks 8 size 2097152 on rank 0
    node-01:1538174:1538174 [0] NCCL INFO NVLS Created Multicast group 0x... nranks 8 size 2097152 on rank 0
    node-01:1538174:1538174 [0] NCCL INFO NVLS rank 0 (dev 0) alloc done, ucptr 0x... ucgran 2097152 mcptr 0x... mcgran 2097152 ucsize 2097152 mcsize 2097152 (inputsize 24576)

Logging to Files
================

For multi-process jobs, logging to the terminal can be overwhelming. It is advisable to use ``NCCL_DEBUG_FILE`` to write logs to separate files per process and hostname, making it easier to isolate and analyze issues on specific ranks:

.. code:: shell

    NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log ./my_app

Format specifiers:

- ``%h`` - Replaced with the hostname
- ``%p`` - Replaced with the process ID (PID)

This creates files like ``/tmp/nccl_node-01_12345.log`` for each process.

Note: Ensure the filename pattern is unique across all processes to avoid file corruption.

Timestamp Configuration
=======================

NCCL log messages can include timestamps for timing analysis.

Timestamp Format
----------------

Use ``NCCL_DEBUG_TIMESTAMP_FORMAT`` to customize the timestamp format (uses strftime syntax):

.. code:: shell

    # Default format: [YYYY-MM-DD HH:MM:SS]
    NCCL_DEBUG=INFO NCCL_DEBUG_TIMESTAMP_LEVELS=INFO ./my_app

    # Include milliseconds
    NCCL_DEBUG=INFO NCCL_DEBUG_TIMESTAMP_LEVELS=INFO NCCL_DEBUG_TIMESTAMP_FORMAT="[%F %T.%3f] " ./my_app

    # Disable timestamps
    NCCL_DEBUG=INFO NCCL_DEBUG_TIMESTAMP_LEVELS=INFO NCCL_DEBUG_TIMESTAMP_FORMAT="" ./my_app

Timestamp Levels
----------------

Control which log levels include timestamps using ``NCCL_DEBUG_TIMESTAMP_LEVELS``:

.. code:: shell

    # Add timestamps to WARN, INFO, and TRACE
    NCCL_DEBUG=INFO NCCL_DEBUG_TIMESTAMP_LEVELS=WARN,INFO,TRACE ./my_app

    # Timestamps on everything except TRACE
    NCCL_DEBUG=TRACE NCCL_DEBUG_TIMESTAMP_LEVELS=^TRACE ./my_app

By default, only WARN messages include timestamps.

Common Debugging Scenarios
==========================

Diagnosing Initialization Hangs
-------------------------------

If your application hangs during NCCL initialization:

.. code:: shell

    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET ./my_app

Look for:

- Bootstrap connection issues
- Network interface selection problems
- Rank synchronization failures

Investigating Network Issues
----------------------------

For network-related problems (timeouts, connection failures):

.. code:: shell

    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET ./my_app

Look for:

- Which network interfaces and devices are selected
- Connection establishment messages
- Error messages from the network transport

Understanding Topology Detection
--------------------------------

To see how NCCL detects and uses the system topology:

.. code:: shell

    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH ./my_app

This shows:

- Detected GPUs and their interconnects
- NVLink and PCIe topology
- Network device locality

Debugging Performance Issues
----------------------------

For performance analysis, enable tuning information:

.. code:: shell

    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING ./my_app

This shows:

- Selected algorithms (Ring, Tree, etc.)
- Protocol choices (Simple, LL, LL128)
- Channel and thread configurations

Tracing NCCL API Calls
----------------------

To see every NCCL function call:

.. code:: shell

    NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=CALL ./my_app

Full Debugging Session
----------------------

For comprehensive debugging, capture everything to a separate file per process:

.. code:: shell

    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log ./my_app

.. highlight:: c++
