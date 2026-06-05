.. py:currentmodule:: nccl.core

*************************
Device Communicator Setup
*************************

Host-side methods and resources for creating an NCCL device communicator.
The device-side communication primitives themselves are available only
from CUDA kernels and are documented under the C device API
(:doc:`../../api/device`); this page covers what the Python (host) side
exposes for bootstrapping them. The configuration object passed to
:py:meth:`Communicator.create_dev_comm` is documented in
:doc:`../configuration`.

create_dev_comm
===============
.. automethod:: Communicator.create_dev_comm

GIN type enums
==============

GPU Interconnect Network (GIN) enums describing what device-side network
transport is available on a communicator and which connection topology
the user requires.

NcclGinType
-----------
.. autoclass:: NcclGinType
   :members:

NcclGinConnectionType
---------------------
.. autoclass:: NcclGinConnectionType
   :members:
