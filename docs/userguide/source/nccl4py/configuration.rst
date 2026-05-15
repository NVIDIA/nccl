.. py:currentmodule:: nccl.core

*************
Configuration
*************

Configuration objects passed to communicator creation methods, plus the
flag enums they consume.

NCCLConfig
==========

Used by :py:meth:`Communicator.init`, :py:meth:`Communicator.split`,
:py:meth:`Communicator.shrink`, and :py:meth:`Communicator.grow`.

.. autoclass:: NCCLConfig

NCCLDevCommRequirements
=======================

Used by :py:meth:`Communicator.create_dev_comm`.

.. autoclass:: NCCLDevCommRequirements

CTAPolicy
=========

.. autoclass:: CTAPolicy
   :members:
   :exclude-members: Default, Efficiency, Zero
