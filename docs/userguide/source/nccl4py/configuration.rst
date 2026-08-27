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

NcclHostCftMode
---------------

Value of :py:attr:`NCCLConfig.host_cft_mode`.

.. autoclass:: NcclHostCftMode
   :members:

NCCLCollConfig
==============

Accepted as the ``config`` argument of every collective on
:py:class:`Communicator`. Fields left unset fall back to the communicator's
value for the resource knobs, and to NCCL's own default otherwise.

.. autoclass:: NCCLCollConfig
   :members:

VendorOption
------------
.. autoclass:: VendorOption
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

NcclCftCap
----------

Bitmask value of :py:attr:`NCCLDevCommRequirements.cft_caps`.

.. autoclass:: NcclCftCap
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
