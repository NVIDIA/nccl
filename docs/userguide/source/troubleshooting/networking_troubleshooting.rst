##########################
Networking Troubleshooting
##########################

.. highlight:: shell

*****************
Networking issues
*****************

IP network interfaces
=====================

NCCL auto-detects which network interfaces to use for inter-node communication.
If some interfaces are in the UP state but are not able to communicate between
nodes, NCCL may try to use them anyway and therefore fail during the init
functions or even hang.

For information about how to specify which interfaces to use, see the
Environment Variables chapter, particularly ``NCCL_SOCKET_IFNAME``.

IP ports
========

NCCL opens TCP ports to connect processes together and exchange connection
information. To restrict the range of ports used by NCCL, one can set the
``net.ipv4.ip_local_port_range`` property of the Linux kernel.

This example restricts NCCL ports to ``50000-51000``:

.. code:: shell

  echo 50000 51000 > /proc/sys/net/ipv4/ip_local_port_range

Or, to make this permanent, add a line to ``/etc/sysctl.conf``:

.. code:: shell

  echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf

Restricting the port range can be useful to open a corresponding range in the
firewall, for example on Google Cloud:

.. code:: shell

  gcloud compute --project=myproject firewall-rules create ncclnet0-ingress --direction=INGRESS --priority=1 --network=ncclnet --action=ALLOW --rules=tcp:50000-51000,22,1024-1039 --destination-ranges=0.0.0.0/0 --target-tags=ncclnet

NIC-level diagnostics
=====================

Before running NCCL over InfiniBand, it is often useful to run low-level
network tests to confirm that the ports are up and operating in the expected
configuration (link layer, link rate, and active state). The ``ibstat`` and
``ibstatus`` tools can be used for a quick sanity check:

.. code:: shell

  # Per-port summary view.
  ibstatus

  # Detailed per-HCA and per-port view.
  ibstat

Look for ports reporting a healthy link, for example:

* Port state is ``Active`` (not ``Down``, ``Init``, or ``Armed``)
* Physical state indicates link up (often ``LinkUp``)
* The expected link layer is in use (for example, ``InfiniBand`` versus
  ``Ethernet`` for RoCE setups) and is consistent across all NICs used for
  transport
* The reported ``Rate`` matches what you expect for your fabric, with no
  unexpected downshift

Bandwidth testing with ``ib_write_bw``
--------------------------------------

If no obvious network issues are present, ``ib_write_bw`` can be used to test
bandwidth between nodes.

Choose two compute nodes to act as server and client. On the server node run:

.. code:: shell

  ib_write_bw -d <device> -a

On the client node run:

.. code:: shell

  ib_write_bw -d <device> <server_hostname_or_ip> -a

Replace ``<device>`` with the HCA device name (for example, ``mlx5_0``) shown
in ``ibstat``.

When troubleshooting GPUDirect RDMA, it is often useful to compare host-memory
results with GPU-memory results. Depending on how ``perftest`` was built, the
GPU-memory options may include commands such as:

.. code:: shell

  # GPU memory path
  ib_write_bw -d mlx5_0 --use_cuda=<gpu_id> <server_hostname_or_ip> -a

  # DMA-BUF path, when supported by the kernel, driver, and perftest build
  ib_write_bw -d mlx5_0 --use_cuda=<gpu_id> --use_cuda_dmabuf <server_hostname_or_ip> -a

Platforms such as GB300 may also expose additional data-direct modes in certain
``perftest`` builds. Check ``ib_write_bw --help`` on your system for the exact
GPU-memory options supported by that build before comparing results.

Latency testing with ``ib_write_lat``
-------------------------------------

Bandwidth can look healthy while tail latency is still poor. If the issue looks
like stalls, long tails, or intermittent slowdowns, also measure point-to-point
latency:

.. code:: shell

  # Server node
  ib_write_lat -d <device>

  # Client node
  ib_write_lat -d <device> <server_hostname_or_ip>

High or unstable latency, especially if bandwidth also fluctuates, can indicate
congestion, retransmissions, or a degraded link.

RDMA protocol statistics
------------------------

``rdma statistic`` can help identify retransmissions or packet-loss-like
symptoms at the RDMA layer:

.. code:: shell

  rdma statistic

Look for counters such as ``rnr_nak_retry_err``, ``packet_seq_err``,
``implied_nak_seq_err``, or ``local_ack_timeout_err`` increasing while a test
is running. Non-zero or growing values here often indicate loss, retries, or
other transport-level problems.

Link health with ``mlxlink``
----------------------------

If the issue looks like degraded PCIe, C2C, or physical port health, check the
link directly with ``mlxlink``:

.. code:: shell

  mlxlink -d <mst_device>

Use this to confirm the expected link width, link speed, error state, and
overall port health. It is especially useful when bandwidth or latency has
degraded without an obvious software configuration problem.

The next sections cover network troubleshooting steps specific to InfiniBand
and RoCE fabrics.

InfiniBand
==========

Subnet Manager
--------------

InfiniBand fabrics require a Subnet Manager (SM) to be running. Check SM status
with:

.. code:: shell

  sudo sminfo

If ``sminfo`` fails or shows no SM, ensure the SM is running on at least one
node (for example, the ``opensm`` service).

Port error counters
-------------------

Use ``perfquery`` to check for link errors, symbol errors, or packet discards
that may indicate cable or switch issues:

.. code:: shell

  sudo perfquery -x <lid>

Non-zero values in error counters, especially ``SymbolErrorCounter``,
``LinkErrorRecoveryCounter``, and ``LinkDownedCounter``, may indicate hardware
or cabling problems.

Connectivity testing
--------------------

If ``ibstat`` and ``ibstatus`` do not indicate issues but a connectivity
problem is suspected, ``ibping`` can be used to test connectivity between
nodes. ``ibping`` requires running a server on the remote node first:

.. code:: shell

  # On the remote node, start the ibping server.
  sudo ibping -S

  # On the local node, ping the server using its LID.
  sudo ibping <lid from ibstat>

Comprehensive diagnostics
-------------------------

For more thorough fabric-wide diagnostics, which usually require SM access, use:

.. code:: shell

  sudo ibdiagnet

This checks topology, routing, and reports errors across the entire fabric.

For example, if NCCL works on some rails but repeatedly fails or slows down on
one specific rail, run ``ibdiagnet`` and inspect the generated report for
missing links, bad ports, or routing inconsistencies affecting the switch ports
connected to that rail before retrying the NCCL job.

Troubleshooting NCCL errors
---------------------------

A common issue seen with InfiniBand is the library not being able to register
sufficient pinned memory. In such cases you may see an error like:

.. code:: shell

  NCCL WARN Call to ibv_create_qp failed

or:

.. code:: shell

  NCCL WARN Call to ibv_reg_mr failed

The solution is to remove the user limits on registering pinned memory. This
can be done by adding these lines:

.. code:: shell

  * soft memlock unlimited
  * hard memlock unlimited

to the ``/etc/security/limits.conf`` configuration file or equivalent on your
Linux distribution. Note that changes to ``limits.conf`` typically require the
user to log out and back in, reboot, or, for Slurm clusters, ensure that jobs
are launched with the updated limits. Verify the new limits with ``ulimit -l``.

RDMA over Converged Ethernet (RoCE)
===================================

RoCE fabrics require different diagnostic tools than InfiniBand to diagnose
inter-node problems.

Port error counters
-------------------

You can use ``ethtool`` to check port counters:

.. code:: shell

  ethtool -S <nic_name>

Check for errors, drops, pause frames, or PFC-related counters. Non-zero values
in these counters may indicate hardware problems, cabling issues, or lossless
fabric misconfiguration.

Congestion indicators
---------------------

RoCE performance problems are often tied to congestion control or lossless
fabric settings rather than outright link failure. In addition to ``ethtool``,
inspect the switch and NIC counters that your environment exposes for PFC, ECN,
CNP, or queue drops. If these counters increase while latency tails grow or
throughput becomes unstable, investigate congestion control and lossless-fabric
configuration before changing NCCL.

Connectivity testing
--------------------

If port-level checks do not indicate any issues but a connectivity problem is
still suspected, ``rping`` can be used to test connectivity between nodes. Note
that ``rping`` requires a server to be running on the remote node first:

.. code:: shell

  # On the remote node, start the rping server.
  rping -s -a <ip_address_of_the_server_nic> -V -C 10

  # On the local node, ping the server using its IP.
  rping -c -a <ip_address_of_the_server_nic> -S <ip_address_of_the_client_nic> -V -C 10

Troubleshooting NCCL errors
---------------------------

A common issue seen with RoCE is the incorrect GID index being selected for the
RoCE v2 NICs. This can result in the following error:

.. code:: shell

  NCCL WARN Call to ibv_modify_qp failed with error Invalid argument

With NCCL 2.21 and later the GID index is dynamically selected, but with prior
versions the user would need to run:

.. code:: shell

  show_gids

and then set ``NCCL_IB_GID_INDEX`` to the GID index for the RoCE v2 GID. With
NCCL 2.21 and later releases, this environment variable should *not* be set.

Users may also need to set ``NCCL_IB_TC`` when using RoCE-based fabrics. Refer
to your vendor's documentation for the values this should be set to.
