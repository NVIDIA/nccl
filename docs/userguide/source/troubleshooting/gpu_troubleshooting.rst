###################
GPU troubleshooting
###################

.. highlight:: shell

**********
GPU Direct
**********

NCCL heavily relies on GPU Direct for inter-GPU communication. This refers to the ability for a GPU to directly
communicate with another device, such as another GPU or a network card, using direct point-to-point PCI messages.

Direct point-to-point PCI messages can fail or perform poorly for a variety of reasons, like missing components,
a bad configuration of a virtual machine or a container, or some BIOS settings.

GPU-to-GPU communication
========================

**Peer-to-peer GPU memory access.** For GPU-to-GPU traffic, NCCL favors *peer-to-peer* transport when CUDA reports that GPUs can access each other's memory directly (typically over NVLink, or over PCIe when the topology and driver allow it).

**Checking peer-to-peer GPU memory access.** You can use ``nvidia-smi topo -p2p <capability>`` to print a matrix of P2P status between GPU pairs. The ``<capability>`` value is ``p`` for PCIe and ``n`` for NVLink.


For example, on a healthy PCI Express-based 8-GPU system with full peer-to-peer GPU memory access, the matrix from ``nvidia-smi topo -p2p p`` typically looks like the example below:

.. code-block:: text

                GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
      GPU0      X       OK      OK      OK      OK      OK      OK      OK
      GPU1      OK      X       OK      OK      OK      OK      OK      OK
      GPU2      OK      OK      X       OK      OK      OK      OK      OK
      GPU3      OK      OK      OK      X       OK      OK      OK      OK
      GPU4      OK      OK      OK      OK      X       OK      OK      OK
      GPU5      OK      OK      OK      OK      OK      X       OK      OK
      GPU6      OK      OK      OK      OK      OK      OK      X       OK
      GPU7      OK      OK      OK      OK      OK      OK      OK      X

If peer-to-peer GPU memory access does not work as expected with NCCL even when ``nvidia-smi topo -p2p`` shows ``OK`` for peer access on the GPU pairs you use, one common cause is **PCI Access Control Services (ACS)**; see :ref:`troubleshooting_acs`.

**CUDA sample.** The `cuda-samples <https://github.com/nvidia/cuda-samples>`__ repository includes a useful program for checking peer-to-peer GPU memory access. Follow the instructions there to build the sample.

**simpleP2P** (``Samples/0_Introduction/simpleP2P``) checks the transferred data after each transfer to confirm that the copy completed successfully. A successful run prints additional lines earlier in the log; the excerpt below is truncated and shows only the final lines:

.. code-block:: text

  ⋮
  Preparing host buffer and memcpy to GPU0...
  Run kernel on GPU1, taking source data from GPU0 and writing to GPU1...
  Run kernel on GPU0, taking source data from GPU1 and writing to GPU0...
  Copy data back to host from GPU0 and verify results...
  Disabling peer access...
  Shutting down...
  Test passed

The last line of the run should read ``Test passed``.

To measure the available bandwidth between GPUs, we recommend using ``nvbandwidth`` rather than the CUDA sample above. Download and build it following the instructions at https://github.com/NVIDIA/nvbandwidth.

**Disabling peer-to-peer GPU memory access for testing.** To see whether a problem is related to the P2P transport, compare runs with peer-to-peer GPU memory access disabled or restricted using :ref:`env_NCCL_P2P_DISABLE` and :ref:`env_NCCL_P2P_LEVEL` (see the :doc:`../env` chapter).

GPU-to-NIC communication
========================

GPUs can also communicate directly with network cards using GPU Direct RDMA (GDRDMA). This requires having compatible
network cards and drivers, plus loading an extra kernel module called ``nvidia-peermem``.
The ``nvidia-peermem`` module is now supplied with the CUDA drivers, however it must be loaded on each node boot with:

.. code:: shell

 sudo modprobe nvidia-peermem

If ``sudo`` is not accessible an alternative way to verify the module has been loaded is to run:

.. code:: shell

  lsmod | grep nvidia-peermem

GDRDMA can also be enabled by using the DMA-BUF feature of recent Linux kernels combined with the open source Nvidia GPU driver.
In this case, NCCL will automatically detect and enable DMA-BUF so the nvidia-peermem module will not be necessary.


.. _troubleshooting_acs:

PCI Access Control Services (ACS)
=================================

**Baremetal systems**

IO virtualization (also known as VT-d or IOMMU) can interfere with GPU Direct by redirecting all PCI point-to-point
traffic to the CPU root complex, causing a significant performance reduction or even a hang. You can check
whether ACS is enabled on PCI bridges by running:

.. code:: shell

  sudo lspci -vvv | grep ACSCtl

If lines show "SrcValid+", then ACS might be enabled. Looking at the full output of lspci, one can check if
a PCI bridge has ACS enabled.

.. code:: shell

  sudo lspci -vvv

If PCI switches have ACS enabled, it needs to be disabled. On some systems this can be done from the BIOS
by disabling IO virtualization or VT-d. For Broadcom PLX devices, it can be done from the OS but needs to
be done again after each reboot.

Use the command below to find the PCI bus IDs of PLX PCI bridges:

.. code:: shell

  sudo lspci | grep PLX

Next, use setpci to disable ACS with the command below, replacing 03:00.0 by the PCI bus ID of each PCI bridge.

.. code:: shell

  sudo setpci -s 03:00.0 ECAP_ACS+0x6.w=0000

Or you can use a script similar to this:

.. code:: shell

  for BDF in `lspci -d "*:*:*" | awk '{print $1}'`; do
    # skip if it doesn't support ACS
    sudo setpci -v -s ${BDF} ECAP_ACS+0x6.w > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      continue
    fi
    sudo setpci -v -s ${BDF} ECAP_ACS+0x6.w=0000
  done

**Virtual machines**

Virtual machines require ACS to function, hence disabling ACS is not an option. To run with maximum
performance inside virtual machines, ATS needs to be enabled in network adapters.

******************
Topology detection
******************

NCCL relies on /sys to discover the PCI topology of GPUs and network cards. When running inside a virtual
machine or container, make sure /sys is properly mounted. Having /sys expose a virtual PCI topology can
result in sub-optimal performance.
