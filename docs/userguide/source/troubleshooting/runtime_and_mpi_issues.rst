######################
Runtime and MPI issues
######################

.. highlight:: shell

******
Errors
******

NCCL calls may return a variety of return codes. Ensure that the return codes are always equal to ncclSuccess. If any call fails and returns a value different from ncclSuccess, setting NCCL_DEBUG to “WARN” will make NCCL print an explicit warning message before returning the error.

Errors are grouped into different categories.

* ncclUnhandledCudaError and ncclSystemError indicate that a call to an external library failed.
* ncclInvalidArgument and ncclInvalidUsage indicate there was a programming error in the application using NCCL.

In either case, refer to the NCCL warning message to understand how to resolve the problem.

*************
Memory issues
*************

.. _cuMem_host_allocations:

Shared memory
=============

To communicate between processes and even between threads of a process, NCCL creates shared memory segments,
traditionally
in /dev/shm. The operating system’s limits on these resources may need to be increased accordingly. Please see your
system’s documentation for details.

If insufficient shared memory is available, NCCL will fail to initialize. Running with NCCL_DEBUG=WARN
will show a message similar to this:

.. code:: shell

 NCCL WARN Error: failed to extend /dev/shm/nccl-03v824 to 4194660 bytes

**Docker**

In particular, Docker containers default to limited shared and pinned memory resources. When using NCCL inside a
container, please make sure to adjust the shared memory size inside the container, for example by adding the following
arguments to the docker launch command line:

.. code:: shell

 --shm-size=1g --ulimit memlock=-1

**Systemd**

When running jobs using mpirun or SLURM, systemd may remove files in shared memory when it detects that the
corresponding user is not logged in, in an attempt to clean up old temporary files. This can cause NCCL to crash
during init with an error like:

.. code:: shell

 NCCL WARN unlink shared memory /dev/shm/nccl-d5rTd0 failed, error: No such file or directory

Given mpirun and SLURM jobs can run on the node without the user being seen as logged in by systemd, system administrators need
to disable that clean-up mechanism, which can be performed by SLURM epilogue scripts instead. To do this, the following
line needs to be set in /etc/systemd/logind.conf:

.. code:: shell

 RemoveIPC=no

Once updated, the daemons should be restarted with:

.. code:: shell

 sudo systemctl restart systemd-logind

**cuMem host allocations**

Starting with version 2.23, NCCL supports an alternative shared memory mechanism using cuMem host allocations. From
NCCL 2.24, if CUDA driver >= 12.6 and CUDA runtime >= 12.2, it is enabled by default in favor of /dev/shm.

However, cuMem host allocations rely on correctly configured and working NUMA support, which may not be available in
some VM and containerization scenarios. In particular, Docker by default disables NUMA support (it can be enabled by
invoking Docker with ``--cap-add SYS_NICE``). From version 2.26.5, NCCL checks if cuMem host allocations work and, if
needed, automatically falls back to the /dev/shm code. In prior versions, the same outcome can be achieved by manually
specifying ``NCCL_CUMEM_HOST_ENABLE=0``. We still recommend configuring the underlying system to ensure that cuMem host
allocations work, as they provide improved reliability during communicator aborts.

cuMem host allocations may fail on systems without CUDA P2P connectivity if CUDA driver version prior to 13.0 is being
used. Furthermore, `CUDA Forward Compatibility
<https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html>`_ feature can affect NCCL's ability to
accurately determine the current driver version, resulting in cuMem host allocations being enabled on older drivers than
intended. We continue to investigate additional mechanisms to detect such circumstances; in the meantime, use
``NCCL_CUMEM_HOST_ENABLE=0`` to deactivate this feature if it causes issues.

Stack size
==========

NCCL's graph search algorithm is highly recursive and, especially on MNNVL
systems where many ranks are reachable via CUDA P2P, may temporarily require
more than 2 MB of thread stack during communicator creation. While the default
Linux stack size limit (8 MB) is known to be sufficient, we've seen crashes
if the limit is changed to ``unlimited``. Due to an idiosyncrasy of GNU libc
(see the man page of ``pthread_create(3)``), such a setting results in a
*decrease* of the stack size of NCCL's background threads to just 2 MB,
which may not be sufficiently large. Use ``ulimit -s`` in bash to print the
current limit; if needed, reset it to 8192 KB using ``ulimit -s 8192`` (one
also needs to ensure that the new setting is propagated to other nodes when
launching a multi-node NCCL job). Starting with version 2.28, NCCL queries the
default stack size for newly launched threads and, if necessary, changes it to
a safe value for the current job. We still recommend that users on affected
systems attempt to get the system-wide setting fixed as -- however well
intentioned -- it is a potentially serious misconfiguration that could have
negative effects extending beyond NCCL jobs.

Unified Memory (UVM)
====================

Starting with version 2.23, NCCL utilizes CUDA memory pools to optimize graph capturing. This feature relies on UVM
being available. While UVM may not be on by default in some virtual machine (VM) setups, it can typically be enabled through a
configuration change.

****************
File Descriptors
****************

NCCL uses a considerable number of file descriptors when running at scale, so the limits may need to be raised.  E.g., a
144-rank job using 16 GIN contexts may require over 32K file descriptors per process.

There is the system-wide limit:

.. code:: shell

 cat /proc/sys/fs/file-max

Default values in the millions are common, and systemd may set it even higher.  If, however, the limit has been
artificially lowered (e.g., by a file under ``/etc/sysctl.d/``), then it may need to be increased again:

.. code:: shell

 sysctl -w fs.file-max=2097152

There is also the per-process limit that can be queried using ``ulimit -n``.  To raise it permanently, create a
new file under ``/etc/security/limits.d/`` (or edit an existing one), adding a line such as:

.. code:: shell

 * - nofile 131072

This sets both the soft and hard limit for all users to 128K.

Note that raising the system-wide limit or the per-process hard limit needs to be done by the system administrator.

***
MPI
***

Before running NCCL with MPI (e.g. ``mpirun <my_application>``), running a simple MPI test can help verify whether the nodes are able to communicate properly.

You can do this in two steps. First, make sure an application can be launched in parallel:

.. code:: shell

 # Open MPI-based implementations:
 mpirun -np <number of processes> -N <processes per node> "hostname"

 # MPICH-based implementations:
 mpirun -np <number of processes> -ppn <processes per node> "hostname"


Second, make sure MPI can be initialized and run a simple reduction:

.. code:: shell

 wget https://raw.githubusercontent.com/pmodels/mpich/main/examples/cpi.c
 mpicc -o cpi cpi.c
 mpirun -np <number of processes> -N <processes per node> ./cpi


Open MPI based MPIs (e.g. NVIDIA HPC-X)
=======================================

Many NCCL-based applications are compiled with MPI to utilize its parallel launcher and broadcast mechanisms during startup. In cluster environments, if MPI is not correctly configured, the ``mpirun`` command may fail to start applications, hang, or produce errors. The following guidelines will help you troubleshoot common MPI-related startup and connectivity issues. These settings assume an environment in which variables are automatically forwarded to each MPI rank (e.g. SLURM cluster). If you are unsure you can explicitly forward the variables through ``mpirun -x VARIABLE_NAME=<variable_value>`` instead of ``export VARIABLE_NAME=<variable_value>``.

These settings will not have any impact on NCCL performance, but if MPI is used frequently for communications, then application performance may be impacted.

Network interface selection
---------------------------

If the application hangs at startup or displays a segmentation fault in ``libmpi.so``, MPI may be selecting an incorrect network interface. You can list active and connected interfaces with:

.. code:: shell

   ip -br link | grep LOWER_UP | grep ' UP '

Usually, only a subset of interfaces (such as ``eth*``, ``en*``, or ``ib*``) are connected to the network. Loopback (``lo``) and container-related interfaces are typically not suitable. If your administrator has specified ``NCCL_SOCKET_IFNAME``, use the same interface with MPI by setting:

.. code:: shell

   export OMPI_MCA_btl_tcp_if_include=<interface-name>

Alternatively, to exclude interfaces that are usually not connected to the network (used for loopback or containers):

.. code:: shell

   export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,virbr0

Note: Do not use include and exclude options simultaneously.

PMIx Data Store selection
-------------------------

There has been an issue in the past (see https://github.com/open-mpi/ompi/issues/7516) with a PMIx component in Open
MPI. This has since been fixed, but it can still occur if your MPI stack is based on an older version. If the
application reports an error similar to:

.. code:: shell

   PMIX ERROR: ERROR in file gds_ds12_lock_pthread.c

You can force a different GDS component through ``export PMIX_MCA_gds=hash``.

UCX and HPC-X considerations
----------------------------

HPC-X commonly utilizes the Unified Communication X (UCX) library. If you encounter UCX warnings such as:

.. code:: shell

   UCX  WARN  network device 'XXX' is not available, please use one or more of: YYY, ...

set the device explicitly:

.. code:: shell

   export UCX_NET_DEVICES=YYY

For UCX error messages like:

.. code:: shell

   UCX  ERROR   no active messages transport to <no debug data>: Unsupported operation
   Error: Failed to resolve UCX endpoint

try simplifying the UCX transport selection:

.. code:: shell

   export UCX_TLS=self,sm,tcp

If necessary, you can disable UCX components and revert to basic TCP communication:

.. code:: shell

   export OMPI_MCA_pml=^ucx
   export OMPI_MCA_coll_hcoll_enable=0
   export OMPI_MCA_coll=^ucc
   export OMPI_MCA_btl=self,tcp
