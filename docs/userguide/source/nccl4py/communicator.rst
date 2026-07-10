.. py:currentmodule:: nccl.core

************
Communicator
************

The :py:class:`Communicator` class and its methods, organized by lifecycle
stage and operation kind:

- :doc:`communicator/class` — the class itself, its constructor, and
  per-instance properties for identity and device-API capability.
- :doc:`communicator/lifecycle` — creating, splitting, growing, and tearing
  down communicators.
- :doc:`communicator/collectives` — collective communication methods
  (allreduce, broadcast, gather, ...).
- :doc:`communicator/p2p` — point-to-point and signal methods
  (send / recv / signal / wait_signal / put_signal).
- :doc:`communicator/registration` — buffer and window registration for
  zero-copy and RMA.
- :doc:`communicator/device_setup` — host-side bootstrap of a device
  communicator.
- :doc:`communicator/status` — error queries and resource cleanup.

.. toctree::
   :maxdepth: 2

   communicator/class
   communicator/lifecycle
   communicator/collectives
   communicator/p2p
   communicator/registration
   communicator/device_setup
   communicator/status
