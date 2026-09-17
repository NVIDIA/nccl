***********
Diagnostics
***********

.. highlight:: none

As of v2.31, NCCL provides a set of built-in diagnostics that can be helpful in identifying
the source of problems observed when running NCCL applications. The diagnostics fall into two
categories:

* **RAS diagnostics** passively collect and compare state information (GPU, CUDA driver, and NCCL
  configuration) across ranks, without exercising NCCL data paths.
* **Active diagnostics** exercise actual communication paths between GPUs to verify that they
  function correctly.

Ideally NCCL diagnostics should be the first course of action when troubleshooting NCCL problems.

RAS Diagnostics
---------------

RAS diagnostics provide a readiness probe for NCCL jobs by checking the selected GPU, CUDA driver,
and NCCL configuration information across ranks. They are part of the RAS subsystem and are
documented in :ref:`ras_diagnostics`.

Active Diagnostics
------------------

Active diagnostics verify that the communication paths NCCL intends to use actually work, by
exchanging data over them and validating the outcome. They run during communicator initialization,
helping diagnose problems that would otherwise surface only later, as hangs, data corruption, or
hard-to-attribute failures of collective operations. Because the checks run before any application
traffic, a reported failure indicates a problem in the system itself — e.g., in the GPU, the
interconnect, or the driver/system configuration — rather than in the application.

Running Active Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^

Active diagnostics are disabled by default. To run them, set ``NCCL_RUN_DIAGNOSTICS=1`` before
starting the application:

.. code::

  NCCL_RUN_DIAGNOSTICS=1 <application> [arguments]

The diagnostics run at each communicator initialization. The report is printed to the standard
output of the process hosting rank 0 of the communicator; each report line is prefixed with
``NCCL DIAG``. A run with no reported issues might look like:

.. code::

  NCCL DIAG === NCCL Diagnostics ===
  NCCL DIAG [OK]   p2p: verified P2P access in both directions between every GPU pair (56 peer accesses)
  NCCL DIAG NCCL diagnostics completed in 245.3 ms across 8 ranks

Result lines use the following tags:

* ``[OK]`` indicates that a check completed without reporting an issue.
* ``[INFO]`` identifies a condition that may require review, such as a failed verification or a
  check that could not be completed.

For a partial P2P result, the summary reports the number of GPU-to-GPU peer accesses that passed
verification out of the total number tested:

.. code::

  NCCL DIAG [INFO] p2p: only 52/56 GPU-to-GPU peer accesses passed verification

Diagnostics are informational only: a reported issue does not abort communicator initialization.

Available Checks
^^^^^^^^^^^^^^^^

P2P Check
"""""""""

The P2P check verifies direct Peer-to-Peer (P2P) access between GPUs. For every pair of GPUs in the
communicator that NCCL expects to communicate directly (e.g., over NVLink or PCIe), each GPU reads
from and writes to its peer, and the diagnostic validates the transferred data. The summary counts
each direction as one peer access. GPU pairs that communicate through an intermediate GPU rather
than direct peer access are excluded from the summary; their ``reason=indirect`` skip records are
available with ``NCCL_DEBUG=INFO`` and ``NCCL_DEBUG_SUBSYS=INIT``.

When a transfer fails, the report identifies the affected GPU pair and the connection path between
them, narrowing the investigation down to a specific link or device. Conversely, a passing P2P
check makes intra-node or NVLink connectivity an unlikely culprit, directing the investigation
toward other components, such as the network between nodes or the application itself.

Network Bandwidth Check
"""""""""""""""""""""""

The network bandwidth check runs ``ib_write_bw`` from the ``perftest`` package over the physical
InfiniBand devices selected by NCCL. It runs only when the communicator uses a network transport
across at least two hosts and requires ``ib_write_bw`` to be installed on every participating
node. The client endpoint of
each measurement connects to the server endpoint's host name, so host names must be resolvable
between the nodes of the communicator. See :doc:`networking_troubleshooting` for instructions on
running the tool manually.

Nodes (ordered by their first rank) form a ring and are checked pairwise in exactly two phases:
pairs starting at even positions run in phase 0 and pairs starting at odd positions in phase 1
(on an odd ring the wrapping pair is skipped; its nodes are covered by their other neighbors).
Within a pair, every device of the server node that the client node also has anchors one check
against that same-named client device; when the selected NCCL topology enables cross-NIC
communication (``NCCL_CROSS_NIC``), phase 0 instead checks against the client's next device after
the anchor, wrapping around the client's device list (a single cross-device rotation). Ranks
sharing a device are paired by occurrence -- the k-th such rank with the k-th on the paired
node -- and measure the shared device concurrently, splitting its bandwidth; a single ``[INFO]``
line from rank 0 notes when shared devices are present.
Each rank measures only the device of its first channel; when ranks use more devices than the one
tested (e.g. a merged device), a single ``[INFO]`` line from rank 0 notes it.
Every rank still runs at most one tool instance per phase, and within a phase all checks proceed
in parallel. Nodes may differ in rank and device counts; a device without a same-named
counterpart on the paired node anchors no check.

Both phases measure from GPU memory, including DMA-BUF, when both endpoints and the installed
tool support it, and from host memory otherwise. Successful runs are not reported individually;
failed runs are reported per device pair. Each endpoint separately averages the successful same-NIC
and cross-NIC measurements in which it participated; the per-rank averages are then shared across all
ranks, and rank 0 prints the minimum, median, and maximum for each mode (tagged ``[OK]``, or
``[INFO]`` when the mode has outliers or not every scheduled measurement produced a result) and
emits an ``[INFO]``
line for each rank deviating by more than 30 percent from that mode's median, in either direction
(at most 8 such lines per mode; a final line reports the count of any further outliers).
Ranks without a successful measurement
in a mode are excluded from that comparison. Example output is shown below:

.. code::

  node0:24563 NCCL DIAG [OK] net bw: 376.1/391.5/398.2 Gbit/s min/median/max same-nic bw (across 8 ranks) in comm 0x7a3f9b2c41d05e88
  node0:24563 NCCL DIAG [INFO] net bw: 265.1/391.5/398.0 Gbit/s min/median/max cross-nic bw (across 8 ranks) in comm 0x7a3f9b2c41d05e88
  node0:24563 NCCL DIAG [INFO] net bw: cross-nic rank 3 (node0): 265.1 Gbit/s, >30% off median 391.5 Gbit/s in comm 0x7a3f9b2c41d05e88

Each measurement streams 1000 messages of 64 KiB. A healthy link completes a measurement in well
under a second; a severely degraded link runs into the per-tool timeout and is reported as
``tool run timed out``.
The check runs under a fixed overall time allocation, separate from the per-tool timeout that
guards each individual ``ib_write_bw`` run. When that time runs out, all nodes skip the remaining
measurements together and a ``test unable to complete in allocated time`` line reports how many
test phases completed.
