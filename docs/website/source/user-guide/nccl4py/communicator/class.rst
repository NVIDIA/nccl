.. py:currentmodule:: nccl.core

******************
Communicator Class
******************

.. autoclass:: Communicator
   :no-members:
   :special-members: __init__

Properties
==========

:py:attr:`Communicator.properties` returns an :py:class:`NCCLCommProperties`
holding the properties NCCL reports for the communicator, including fields that
have no dedicated accessor. The groups below provide per-field accessors for
the values needed most often.

.. autoattribute:: Communicator.properties

Identity
--------

.. autoattribute:: Communicator.ptr
.. autoattribute:: Communicator.is_valid
.. autoattribute:: Communicator.nranks
.. autoattribute:: Communicator.device
.. autoattribute:: Communicator.rank

Device-API capability
---------------------

These properties reflect the underlying NCCL :c:type:`ncclCommProperties_t`
structure.

.. autoattribute:: Communicator.cuda_dev
.. autoattribute:: Communicator.nvml_dev
.. autoattribute:: Communicator.device_api_support
.. autoattribute:: Communicator.multimem_support
.. autoattribute:: Communicator.gin_type
.. autoattribute:: Communicator.n_lsa_teams
.. autoattribute:: Communicator.host_rma_support
.. autoattribute:: Communicator.railed_gin_type

NCCLCommProperties
------------------

Covers the accessors in both groups above, plus the fields that have none.

.. autoclass:: NCCLCommProperties
   :members:

Teams
=====

A :py:class:`NCCLTeam` names a strided subset of the communicator's ranks.
The members below return the predefined teams; pass one to
:py:class:`TeamRequirement` or to the rank converters. See
:ref:`devapi_teams` for team semantics.

.. autoattribute:: Communicator.team_world
.. autoattribute:: Communicator.team_lsa
.. autoattribute:: Communicator.team_rail
.. autoattribute:: Communicator.team_cft_multimem

.. automethod:: Communicator.team_cft
.. automethod:: Communicator.team_rank_to_world
.. automethod:: Communicator.team_rank_to_lsa

NcclCftTeamMode
---------------
.. autoclass:: NcclCftTeamMode
   :members:
