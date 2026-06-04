###############
Troubleshooting
###############

Use the pages below to narrow the problem down before changing NCCL settings.

* :doc:`troubleshooting/gpu_troubleshooting` covers GPU-to-GPU, GPU-to-NIC, ACS, topology, and multi-node NVLink issues.
* :doc:`troubleshooting/networking_troubleshooting` covers interface selection, low-level fabric checks, latency and bandwidth tests, and InfiniBand or RoCE diagnostics.
* :doc:`troubleshooting/runtime_and_mpi_issues` covers basic error handling, shared-memory and runtime problems, and MPI startup validation.
* :doc:`troubleshooting/performance_and_tuning` covers baseline performance triage and NCCL tuning knobs to try after system checks look healthy.
* :doc:`troubleshooting/logging` covers NCCL logging levels, subsystem filters, output files, and timestamps.
* :doc:`troubleshooting/ras` covers NCCL's built-in RAS subsystem for diagnosing hangs and crashes.

.. toctree::
   :hidden:
   :maxdepth: 1

   troubleshooting/gpu_troubleshooting
   troubleshooting/networking_troubleshooting
   troubleshooting/runtime_and_mpi_issues
   troubleshooting/performance_and_tuning
   troubleshooting/logging
   troubleshooting/ras
