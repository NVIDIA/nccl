.. py:currentmodule:: nccl.core

*********
Utilities
*********

Identifiers, version helpers, error helpers, and the Python-side validation
exception.

UniqueId
========

.. autoclass:: UniqueId
   :members:
   :exclude-members: ptr

get_unique_id
=============
.. autofunction:: get_unique_id

Version
=======

.. autoclass:: Version
   :members:

get_version
===========
.. autofunction:: get_version

get_error_string
================
.. autofunction:: get_error_string

NcclInvalid
===========
.. autoexception:: NcclInvalid
