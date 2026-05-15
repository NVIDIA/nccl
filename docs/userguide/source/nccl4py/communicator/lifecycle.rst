.. py:currentmodule:: nccl.core

******************************
Creation and Lifecycle Methods
******************************

Methods on :py:class:`Communicator` for creation, splitting, growing, and
teardown.

Construction
============

.. automethod:: Communicator.init
.. automethod:: Communicator.init_all
.. automethod:: Communicator.initialize

Splitting and growing
=====================

.. automethod:: Communicator.split
.. automethod:: Communicator.shrink
.. automethod:: Communicator.get_unique_id
.. automethod:: Communicator.grow

Teardown
========

.. automethod:: Communicator.destroy
.. automethod:: Communicator.abort
.. automethod:: Communicator.finalize

Pause and resume
================

.. automethod:: Communicator.revoke
.. automethod:: Communicator.suspend
.. automethod:: Communicator.resume

Flag enums
==========

CommShrinkFlag
--------------

.. autoclass:: CommShrinkFlag
   :members:
   :exclude-members: Default, Abort

CommSuspendFlag
---------------

.. autoclass:: CommSuspendFlag
   :members:
   :exclude-members: Mem
