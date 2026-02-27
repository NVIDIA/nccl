*****************************************************
nccl4py: Pythonic NCCL Communication for GPU Clusters
*****************************************************

.. image:: https://img.shields.io/badge/NVIDIA-black?logo=nvidia
   :target: https://www.nvidia.com/
   :alt: NVIDIA

`nccl4py <https://github.com/NVIDIA/nccl>`_ bridges Python's simplicity with the performance of NVIDIA Collective Communications Library (NCCL), and provides a Pythonic interface to NCCL library's functionality. It enables Python applications to leverage NCCL's GPU-accelerated multi-GPU and multi-node communication capabilities for distributed computing workloads.

nccl4py follows the NCCL SLA. The details of the NCCL SLA is available `here <https://docs.nvidia.com/deeplearning/nccl/sla/index.html>`_.

* `Homepage <https://developer.nvidia.com/nccl>`_
* `Repository <https://github.com/NVIDIA/nccl>`_
* `Documentation <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`_
* `Issue tracker <https://github.com/NVIDIA/nccl/issues>`_

``nccl4py`` is under active development. Feedback and suggestions are welcome!


Installation
============

For CUDA 12.x:

.. code-block:: bash

   pip install nccl4py[cu12]

For CUDA 13.x:

.. code-block:: bash

   pip install nccl4py[cu13]

