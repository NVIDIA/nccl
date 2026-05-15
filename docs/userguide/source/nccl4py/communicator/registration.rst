.. py:currentmodule:: nccl.core

***************************
Memory Registration Methods
***************************

Methods on :py:class:`Communicator` for registering buffers and windows for
zero-copy and RMA operations. The returned handle classes are documented
under :doc:`../memory`.

register_buffer
===============
.. automethod:: Communicator.register_buffer

register_window
===============
.. automethod:: Communicator.register_window

WindowFlag
==========

.. autoclass:: WindowFlag
   :members:
   :exclude-members: Default, CollSymmetric, StrictOrdering
