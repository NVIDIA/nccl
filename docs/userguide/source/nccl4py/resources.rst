.. py:currentmodule:: nccl.core

***********************
Communicator Resources
***********************

Resource handles owned by a :py:class:`Communicator`. They share a common
lifecycle: each handle is tracked by its owning communicator and is
released either explicitly via its ``close()`` method or automatically
when the communicator is destroyed or aborted.

CommResource
============
.. autoclass:: nccl.core.resources.CommResource
   :members:

RegisteredBufferHandle
======================
.. autoclass:: RegisteredBufferHandle
   :members:

RegisteredWindowHandle
======================
.. autoclass:: RegisteredWindowHandle
   :members:

CustomRedOp
===========
.. autoclass:: CustomRedOp
   :members:

DevCommResource
===============
.. autoclass:: DevCommResource
   :members:
   :exclude-members: dev_comm
