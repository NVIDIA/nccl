***
RAS
***

.. highlight:: none

Since NCCL 2.24, the reliability, availability, and serviceability (RAS) subsystem can be used to query the health of
NCCL jobs during execution.  This can help with the diagnosis and debugging of crashes and hangs.  RAS is a low-overhead
infrastructure that NCCL users and developers can use while the application is running.  It provides a global view of
the state of the running application and can aid in the detection of outliers such as unresponsive processes.  With
that information, users can then narrow down on the suspected root cause(s) through other techniques such as interactive
debugging, system log analysis, etc.

Principle of Operation
----------------------

RAS is built into NCCL and launches during NCCL initialization.  It consists of a set of threads (one per process) that
establish connections with each other, forming a network that the RAS threads then use to exchange information and
monitor each other's health.  In a typical configuration, the RAS network traffic (which uses plain TCP/IP sockets on
top of the bootstrap/out-of-band network interface that NCCL uses during initialization) should not compete with the
main NCCL traffic (which utilizes RDMA networking).  RAS is lightweight and should not interfere with the main NCCL job;
as such, it is enabled by default (but see :ref:`env_NCCL_RAS_ENABLE`).

The RAS threads communicate with each other about any changes to the job configuration; they also exchange regular
keep-alive messages.  If a NCCL process crashes or hangs, the RAS threads running on other NCCL processes
learn about it through the RAS network connections to that process being shut down or becoming unresponsive.

RAS Queries
-----------

The RAS threads also listen for client connections on ``localhost``, port ``28028`` (these defaults can be changed using
:ref:`env_NCCL_RAS_ADDR`).  The ``ncclras`` binary client can be used to connect to that socket and query the RAS
subsystem for the current job status, which is then printed to standard output.  The client accepts the ``-h`` and
``-p`` arguments to specify the host name and port, ``-v`` to produce a more verbose output in case of problems, and
``-t`` to specify a different timeout (``5`` seconds by default; `0` disables the timeout).

As the client communication protocol is fully text-based, standard networking tools such as telnet or netcat can be used
instead of the ``ncclras`` binary.  The relevant commands include ``STATUS``, ``VERBOSE STATUS`` (equivalent to the
``ncclras`` client's ``-v`` argument), and ``TIMEOUT <seconds>`` (equivalent to ``-t``); e.g., ``echo verbose status |
nc localhost 28028``.

Irrespective of how the query is submitted, the receiving RAS thread sends back the job summary information as well as
the summary information about all the NCCL communicators; the latter is collected from all the job's processes so, for
jobs experiencing problems or ones that are particularly large, the response may take several seconds to generate.  In
case any issues were encountered, additional information is provided.

Sample Output
-------------

This section contains excerpts of the RAS status output.  Please note that the exact format and scope of the information
being made available varies from release to release; the excerpts are provided for illustrative purposes only.  For a
more machine-friendly format, see :ref:`ras_json` below.

Here's an example output from a job that is progressing normally:

.. code::

  Job summary
  ===========

    Nodes  Processes         GPUs  Processes     GPUs
  (total)   per node  per process    (total)  (total)
        4          8            1         32       32

We've got a job consisting of 32 GPUs (1 GPU per process) running on 4 nodes.

.. code::

  Communicators... (0.00s)
  =============

  Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
      #  in group  per comm  per node  per comm  in group
      0         8         4         1         4        32   RUNNING      OK

The GPUs are split into 8 communicators, 1 GPU per node.  RAS attempts to make the summary output as short as possible
by grouping together objects having the same size and other important properties.

For jobs that are actively communicating during the RAS query, the following output can sometimes be observed:

.. code::

  Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
      #  in group  per comm  per node  per comm  in group
      0         1         4         8        32        32   RUNNING  MISMATCH

The output indicates that there is an inconsistency in the information provided by different communicator ranks.
Additional information is printed underneath (in this case it's in the Warnings section, indicating a potentially lower
severity):

.. code::

  Warnings
  ========

  #0-0 (27a079b828ff1a75) MISMATCH
    Communicator ranks have different collective operation counts
    26 ranks have launched up to operation 6650
    6 ranks have launched up to operation 6649
    Rank 0 -- GPU 0 managed by process 483072 on node 172.16.64.210
    Rank 2 -- GPU 2 managed by process 483074 on node 172.16.64.210
    Rank 3 -- GPU 3 managed by process 483075 on node 172.16.64.210
    Rank 4 -- GPU 4 managed by process 483076 on node 172.16.64.210
    Rank 5 -- GPU 5 managed by process 483077 on node 172.16.64.210
    Rank 7 -- GPU 7 managed by process 483079 on node 172.16.64.210

Communicators are referred to using the ``#<x>-<y>`` identifiers, where ``<x>`` is the group number from the
summary output and ``<y>`` is the communicator number within the group, both starting with 0 (in this example there is
only one (32-GPU) communicator so, unsurprisingly, the identifier is ``#0-0``).  The identifier is followed by a
communicator hash, which is a value that can be found in NCCL's regular debug output as well, and the rank information.
RAS groups together the ranks with the same relevant property (the count of issued collective operations in
this case; starting with NCCL 2.26, this is broken down per collective operation type).  If a group constitutes an
outlier, RAS prints additional information about each group member.  By default
this is done if the group size is at most 25% of the total *and* the group has no more than 10 members; enabling verbose
output relaxes this to under 50% of the total and lifts the group size limit.

The particular case above should not be a cause for concern, as long as the counts increase across repeated queries.
NCCL collectives, being optimized for speed, can easily outpace the RAS collective queries, especially if the size of
the collectives is fairly small.  An application may also exhibit work imbalance, with certain ranks routinely arriving
to the collective operations later than others -- an experience with a particular workload is needed to determine what's
normal and what's not.  However, if the output does not change across subsequent RAS queries, it may indicate that the
communicator is "stuck" for some reason, which could warrant an investigation.

Similar effects can sometimes be observed during communicator initialization or tear-down:

.. code::

  Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
      #  in group  per comm  per node  per comm  in group
      0         1         4       1-2        32        32  FINALIZE  MISMATCH
      1         7         4         1         4        28   RUNNING      OK
      2         1         4         1         4         4      INIT      OK

  [...]

  #0-0 (9e17999afaa87dbb) MISMATCH
    Communicator ranks have different status
    26 ranks have status UNKNOWN
    4 ranks have status RUNNING
    Rank 0 -- GPU 0 managed by process 507285 on node 172.16.64.210
    Rank 8 -- GPU 0 managed by process 1598388 on node 172.16.64.212
    Rank 16 -- GPU 0 managed by process 3500071 on node 172.16.64.213
    Rank 24 -- GPU 0 managed by process 2405067 on node 172.16.64.222
    2 ranks have status FINALIZE
    Rank 4 -- GPU 4 managed by process 507289 on node 172.16.64.210
    Rank 20 -- GPU 4 managed by process 3500075 on node 172.16.64.213

The above snapshot depicts a transitional situation as the initial, 32-GPU communicator is being replaced by eight 4-GPU
communicators (one of which is still initializing, so it is listed separately (group ``#2``) from the already
initialized seven (group ``#1``)).  The
32-GPU communicator (``#0-0``) is being torn down, with two ranks in the middle of `ncclCommFinalize`, four ranks that
have *not* called `ncclCommFinalize` yet, and the remaining 26 ranks "unknown" -- meaning that they didn't provide any
information about that communicator when RAS was collecting data, simply because their call to `ncclCommFinalize` has
already completed so they are in fact no longer that communicator's members (NCCL 2.26 and later print ``NOCOMM``
instead).  Again, as long as the situation is
resolved when the query is repeated, it can be ignored.

Here's an excerpt from an invocation right after artificially creating a problem with one of the job processes:

.. code::

  Communicators... (2.05s)
  =============

  Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
      #  in group  per comm  per node  per comm  in group
      0         1         4       7-8        32        32   RUNNING  INCOMPLETE

  Errors
  ======

  INCOMPLETE
    Missing communicator data from 1 job process
    Process 3487984 on node 172.16.64.213 managing GPU 5

  #0-0 (cf264af53edbe986) INCOMPLETE
    Missing communicator data from 1 rank
    The missing rank: 21

  Warnings
  ========

  TIMEOUT
    Encountered 2 communication timeouts while gathering communicator data

In this case the summary takes a few seconds to generate because RAS waits for the data from the process experiencing
problems (the process is unresponsive -- it was stopped -- but RAS doesn't know it yet).  Repeated queries should be
much faster because once RAS determines that a process is unresponsive, it reconfigures the RAS network to route around
it.

RAS will attempt to reestablish communication with the unresponsive process; if it's unable to do so for 60 seconds, it
will declare the process dead (permanently):

.. code::

  Errors
  ======

  DEAD
    1 job process is considered dead (unreachable via the RAS network)
    Process 3487984 on node 172.16.64.213 managing GPU 5

  #0-0 (cf264af53edbe986) INCOMPLETE
    Missing communicator data from 1 rank
    The missing rank: 21

RAS will simply stop attempting to communicate with such processes over the RAS network anymore, leaving it up to the
user to determine if any additional action is warranted.

.. _ras_json:

JSON Output
-----------

Starting with NCCL 2.28.7, RAS can generate output in JSON format to support machine-parsable metrics collection.

The ``ncclras`` binary gains an additional option ``-f`` followed by an argument: ``text`` or ``json``, with ``text``
being the default.  The equivalent wire-level protocol command is ``SET FORMAT <format>``.  Sample output can be found
below:

.. code::

  {
    "nccl_version": "2.29.1",
    "cuda_runtime_version": 13000,
    "cuda_driver_version": 13000,
    "timestamp": "2025-12-19 13:06:53",
    "communicators_count": 1,
    "communicators": [
      {
        "hash": "0xae94423cfbb2ef4a",
        "secondary_hash": "0xb7e7187447156001:0xb8242ed28a71381e",
        "size": 2,
        "ranks_count": 1,
        "missing_ranks_count": 1,
        "ranks": [
          {
            "rank": 0,
            "host": "172.16.64.245",
            "pid": 1524344,
            "cuda_dev": 0,
            "nvml_dev": 0,
            "status": {
              "init_state": 0,
              "async_error": 0,
              "finalize_called": false,
              "destroy_flag": false,
              "abort_flag": false
            },
            "collective_counts": {
              "Broadcast": 0,
              "Reduce": 0,
              "AllGather": 0,
              "ReduceScatter": 0,
              "AllReduce": 0
            }
          }
        ],
        "missing_ranks": [
          {
            "rank": 1,
            "host": "172.16.64.245",
            "pid": 1524345,
            "cuda_dev": 1,
            "nvml_dev": 1,
            "status": {
              "unresponsive": true,
              "considered_dead": false
            }
          }
        ]
      }
    ],
    "ras": {
      "collection_time_sec": 0.000,
      "timeouts_count": 0
    }
  }

As can be observed, the JSON output is considerably more verbose than the text one (which is optimized for human
consumption).  It is essentially a dump of all raw data collected by RAS; the analysis and interpretation is left to the
consumer.

Most of the fields should be fairly self-explanatory, with the possible exception of:

* ``secondary_hash``: can be used in conjunction with ``hash`` to create a communicator identifier that is guaranteed to
  be unique (which ``hash`` by itself is not, although in practice collisions are highly unlikely).
* ``nvml_dev``: normally the same as ``cuda_dev``, unless `CUDA_VISIBLE_DEVICES` is being used (in which case
  ``cuda_dev`` may not be unique among ranks on a single node, whereas ``nvml_dev`` always is).
* ``init_state``: ``0`` (`ncclSuccess`) if a rank is fully initialized, ``7`` (`ncclInProgress`) if initialization is
  still ongoing; any other value indicates an error.
* ``async_error``: the value that would be returned if the rank called `ncclGetAsyncError`.
* ``unresponsive``: `true` if RAS was unable to reach the rank when collecting data -- this is the most common
  reason why a rank would be considered missing.  There are, however, corner cases during communicator initialization
  and termination when a process may be reachable but does not consider itself to be a member of a given communicator.

Monitoring Mode
---------------

Starting with NCCL 2.29, RAS adds a monitoring mode for real-time status updates.

The ``ncclras`` binary gains an additional option ``-m`` that switches it to monitoring mode.  The equivalent wire-level
protocol command is ``MONITOR``.

When in monitoring mode, the RAS client prints a welcome message and does not terminate until it is interrupted or the
job finishes:

.. code::

  RAS Monitor Mode - watching for peer changes (Ctrl+C to exit)...
  ================================================================

A sample event of interest could be a process being declared dead:

.. code::

   [2025-12-19 13:07:07] PEER_DEAD: Process 1524345 on node 172.16.64.245 managing GPU 1

Monitoring mode can also be used in conjunction with JSON output:

.. code::

  {
    "timestamp": "2025-12-19 13:07:07",
    "group": "LIFECYCLE",
    "event": "PEER_DEAD",
    "peer": {
      "host": "172.16.64.245",
      "pid": 1524345,
      "cuda_devs": [1],
      "nvml_devs": [1]
    },
    "details": ""
  }

Unlike in the previously shown communicator output (where each rank was printed separately), here the entity of concern
is a `process` so ``cuda_devs`` and ``nvml_devs`` need to be arrays (since a process can manage multiple GPUs).

.. highlight:: shell
