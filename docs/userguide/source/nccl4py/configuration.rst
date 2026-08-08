.. py:currentmodule:: nccl.core

*************
Configuration
*************

Configuration objects passed to communicator creation methods and to
individual collectives, plus the flag enums they consume.

NCCLConfig
==========

Used by :py:meth:`Communicator.init`, :py:meth:`Communicator.initialize`,
:py:meth:`Communicator.split`, :py:meth:`Communicator.shrink`, and
:py:meth:`Communicator.grow`. Fields left unset (``None``) remain at NCCL's
internal default; values are validated by the C library when the config is
consumed.

.. autoclass:: NCCLConfig
   :members:

CTAPolicy
=========

.. autoclass:: CTAPolicy
   :members:
   :exclude-members: Default, Efficiency, Zero

NCCLDevCommRequirements
=======================

Used by :py:meth:`Communicator.create_dev_comm`. Fields left unset
(``None``) remain at NCCL's internal default.

.. autoclass:: NCCLDevCommRequirements
   :members:

Requirement entries
===================

The element types of :py:attr:`NCCLDevCommRequirements.teams` and
:py:attr:`NCCLDevCommRequirements.resources`.

TeamRequirement
---------------
.. autoclass:: TeamRequirement
   :members:

LsaBarrierRequirement
---------------------
.. autoclass:: LsaBarrierRequirement
   :members:

GinBarrierRequirement
---------------------
.. autoclass:: GinBarrierRequirement
   :members:

LLA2ARequirement
----------------
.. autoclass:: LLA2ARequirement
   :members:
