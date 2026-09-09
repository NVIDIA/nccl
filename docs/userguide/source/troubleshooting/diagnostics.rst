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
  NCCL DIAG [OK]   p2p: all 56 directed GPU P2P edges verified
  NCCL DIAG NCCL diagnostics completed in 245.3 ms across 8 ranks

Result lines use the following tags:

* ``[OK]`` indicates that a check completed without reporting an issue.
* ``[INFO]`` identifies a condition that may require review, such as a failed verification or a
  check that could not be completed.

Diagnostics are informational only: a reported issue does not abort communicator initialization.

Available Checks
^^^^^^^^^^^^^^^^

P2P Check
"""""""""

The P2P check verifies direct Peer-to-Peer (P2P) communication between GPUs. For every pair of
GPUs in the communicator that NCCL expects to communicate directly (e.g., over NVLink or PCIe),
it transfers data between the two GPUs and validates that the data arrives intact.

When a transfer fails, the report identifies the affected GPU pair and the connection path between
them, narrowing the investigation down to a specific link or device. Conversely, a passing P2P
check makes intra-node or NVLink connectivity an unlikely culprit, directing the investigation
toward other components, such as the network between nodes or the application itself.
