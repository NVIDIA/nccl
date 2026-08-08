.. py:currentmodule:: nccl.core

***********************
Communicator Resources
***********************

Resources owned by a :py:class:`Communicator`. The
:py:class:`~nccl.core.resources.CommResource`
subclasses below are tracked by their owning communicator and released either
explicitly via ``close()`` or automatically when the communicator is destroyed
or aborted.

CommResource
============

Abstract base class of communicator-owned resources; it defines their
``close()`` and ``is_valid`` contract.

.. autoclass:: nccl.core.resources.CommResource
   :members:

RegisteredBufferHandle
======================
.. autoclass:: RegisteredBufferHandle
   :members:
   :inherited-members:

RegisteredWindowHandle
======================
.. autoclass:: RegisteredWindowHandle
   :members:
   :inherited-members:

CustomRedOp
===========
.. autoclass:: CustomRedOp
   :members:
   :inherited-members:

DevCommResource
===============
.. autoclass:: DevCommResource
   :members:
   :inherited-members:
   :exclude-members: dev_comm

Device resource handles
=======================

Handles returned by :py:attr:`DevCommResource.resource_handles` and
:py:meth:`DevCommResource.multimem_handle`. These are views backed by their
owning :py:class:`DevCommResource`, not independently closable resources. They
remain valid only while that resource remains open. Pass one to the device-side
APIs.

MultimemHandle
--------------
.. autoclass:: MultimemHandle
   :no-show-inheritance:

LsaBarrierHandle
----------------
.. autoclass:: LsaBarrierHandle
   :no-show-inheritance:

GinBarrierHandle
----------------
.. autoclass:: GinBarrierHandle
   :no-show-inheritance:

LLA2AHandle
-----------
.. autoclass:: LLA2AHandle
   :no-show-inheritance:
