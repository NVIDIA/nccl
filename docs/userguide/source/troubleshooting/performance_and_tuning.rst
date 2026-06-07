######################
Performance and tuning
######################

.. highlight:: shell

******************
Performance issues
******************

Performance issues may be caused by a variety of factors and may be specific to a particular application or particular hardware. Under these conditions it is important to differentiate a NCCL performance bug from possible configuration or hardware issues.

``nvbandwidth`` (https://github.com/NVIDIA/nvbandwidth). This tool can be used to measure GPU memory bandwidth and GPU-to-GPU bandwidth (via NVLink or PCIe). When compiled with multinode support, it can also measure Multi-node NVLink (MNNVL) bandwidth between nodes.

``nvloom`` (https://github.com/NVIDIA/nvloom). Is an alternative tool to measure GPU memory bandwidth and GPU-to-GPU bandwidth (via NVLink or PCIe). It delivers similar functionality to ``nvbandwidth``, however, it also provides benchmarks for NVLink SHARP systems.

Intra-node communication
========================

By default, ``nvbandwidth`` is compiled to test intra-node communication. Possible issues to look out for are node topology and NVLink issues.
``nvidia-smi topo -m`` shows the intra-node topology, which can be used to determine the expected communication bandwidth between components.
Another check to run is ``nvidia-smi topo -p2p n`` to verify if GPUs can communicate directly with each other over NVLink. With this information ``nvbandwidth`` can be used to verify if the expected bandwidth between individual pairs of GPUs can be achieved (via PCIe or NVLink).

In the case of ``nvloom`` pairwise and multicast tests can be run in order to benchmark local communication:

.. code:: shell

  srun/mpirun -n <number of processes> ./nvloom -s pairwise --sizeMin 1M --sizeMax 4G
  srun/mpirun -n <number of processes> ./nvloom -s multicast --sizeMin 1M --sizeMax 4G

Inter-node communication
========================

To profile inter-node performance, ``nvbandwidth`` must be compiled with multinode support:

.. code:: shell

  cmake -DMULTINODE=1 .
  make

Once compiled, the tool can be used to measure the bandwidth and latency of the network.

.. code:: shell

  srun/mpirun -n <number of processes> ./nvbandwidth -p multinode

This will run tests prefixed with ``multinode`` using given number of processes.

The equivalent test for ``nvloom`` is to run:

.. code:: shell

  srun/mpirun -n <number of processes> ./nvloom -p gpu-to-rack rack-to-rack fabric-stress --sizeMin 1M --sizeMax 4G

Bandwidth reported by ``nvbandwidth`` (or ``nvloom``) lower than the expected peak bandwidth indicates an issue with
inter-GPU communication. One possible cause is that there is no NVLink connection between the GPUs. Please use ``nvidia-smi topo -p2p n`` to verify
if GPUs can communicate directly with each other over NVLink.

To test fabric performance, ``ib_write_bw`` and ``ib_write_lat`` can be used to
measure bandwidth and latency between nodes, as described in
:doc:`networking_troubleshooting`.

If ``nvbandwidth``/``nvloom`` and ``ib_write_bw`` results match the expectations for the hardware but NCCL performance is below expectations, the NCCL configuration might be suboptimal. Check :ref:`optimize_nccl_config` for the guidance on tweaking NCCL configuration.

Interpreting profiler kernel names
==================================

NCCL may use representative device kernel names for multiple algorithm and
protocol combinations to reduce the number of specialized kernels in the
library. As a result, profilers such as Nsight Systems can show a kernel name
with an ``LL`` suffix even when NCCL selected a different protocol for the
collective. Use ``NCCL_DEBUG_SUBSYS=TUNING`` when the exact selected algorithm
and protocol are needed.

Multi-node NVLink (MNNVL) issues
================================

NCCL uses MNNVL for inter-node communication within the same NVLink domain if available. If it is not used and not disabled by the user (i.e.  NCCL_MNNVL_ENABLE is not set to 0) there might be an underlying issue with the service.
To diagnose the issue, you may use Internode Memory Exchange (IMEX) service. You can use ``nvidia-imex-ctl`` utility to check the status of the IMEX domain. NOTE: this requires IMEX daemon to be running on every node of the domain and may potentially require sudo privileges.

.. code:: shell

  nvidia-imex-ctl -H -N

Check the output first for the ``Domain State`` line. If it is not ``UP`` then you should check status of each node in the domain and their connectivity matrices. Some of the common causes of issues may include:

* Node down (status ``UNAVAILABLE`` and connectivity matrix shows ``I/N/D``)
* Driver version mismatch between nodes (status ``READY`` and connectivity matrix shows ``V``)

A healthy IMEX domain should have all nodes in the domain in the ``READY`` state and connectivity matrix showing ``C``.

In some cases NCCL may report this warning:

.. code:: shell

  transport/p2p.cc:XXX NCCL WARN Cuda failure 800 'operation not permitted'

This may indicate that the current user has no write access to IMEX security files located at ``/dev/nvidia-caps-imex-channels/channel*``.
Changing their permissions to allow write access should fix the issue.

In cases when user has no access to IMEX daemon an alternative is to use ``nvidia-smi -q | grep -v GUID | grep -A4 Fabric``
to check the NVLink fabric status as well as verify that the cliqueId is the same across all the nodes in the NVLink domain.

.. _optimize_nccl_config:

Tuning NCCL configuration
=========================

NCCL is tuned to run optimally on a wide range of systems and re-tuned with newer release. However, there are some edge cases where a system can benefit from different settings. The most common tuning parameters are listed below.

NOTE: In general we discourage the use of these variables in production since a tuning gain in one benchmark situation can lead to suboptimal settings elsewhere.

.. code:: shell

  NCCL_MIN_CTAS, NCCL_MAX_CTAS - Increasing the number of CTAs will consume more GPU resources but possibly increase throughput.

  NCCL_CHUNK_SIZE - Controls the size of messages sent through the network for ncclSend/ncclRecv and AlltoAll operations.
                    Increasing this number may help improve bandwidth in latency-bound cases.

  NCCL_IB_QPS_PER_CONNECTION - This controls the number of QPs per connection. The default value is 1. However, on
                               systems with ECMP routing enabled or multiple ports per NIC, increasing this value
                               can improve path diversity on the network and increase throughput.

  NCCL_CROSS_NIC - This controls whether NCCL allows rings and trees to use different NICs, causing inter-node
                   communication to use different NICs on different nodes. Forcing cross-NIC communication may
                   improve performance in unoptimized rail configurations but may create congestion on other
                   networks. The default value is 2.

RoCE considerations
--------------------

On RoCE fabric, using multiple QPs per connection is often necessary to achieve optimal performance.

Process/thread affinity
=======================
Incorrect process and thread placement can have a serious performance impact. The NCCL_IGNORE_CPU_AFFINITY environment variable will let NCCL assign CPU affinity of the threads it creates based on GPU affinity.
However, process placement is in the hands of the user and depending on the node architecture the requirements may vary. Users can set the CPU affinity via ``numactl``, OpenMPI's ``--bind-to``, the ``--cpu-bind`` option of ``srun`` or machine file options. A general rule is to rely on the information provided by ``nvidia-smi topo -m`` and spread out processes based on GPU-CPU affinity reported by the tool.
