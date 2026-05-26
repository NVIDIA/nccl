.. py:currentmodule:: nccl.core

*************
Configuration
*************

Configuration objects passed to communicator creation methods, plus the
flag enums they consume.

NCCLConfig
==========

Used by :py:meth:`Communicator.init`, :py:meth:`Communicator.split`,
:py:meth:`Communicator.shrink`, and :py:meth:`Communicator.grow`. Fields
left unset (``None``) remain at NCCL's internal default; values are
validated by the C library when the config is consumed.

.. autoclass:: NCCLConfig
   :members:

NCCLDevCommRequirements
=======================

Used by :py:meth:`Communicator.create_dev_comm`. Fields left unset
(``None``) remain at NCCL's internal default.

.. autoclass:: NCCLDevCommRequirements
   :members:

CTAPolicy
=========

.. autoclass:: CTAPolicy
   :members:
   :exclude-members: Default, Efficiency, Zero
