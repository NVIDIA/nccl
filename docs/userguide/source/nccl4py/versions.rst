.. py:currentmodule:: nccl

********
Versions
********

NCCL4Py exposes top-level helpers to inspect the installed NCCL stack:
``nccl4py`` itself plus the native libraries ``libnccl.so`` and
``libnccl_ep.so``.

.. code-block:: python

    import nccl
    nccl.show_versions()      # human-readable block to stdout
    v = nccl.get_version()    # programmatic snapshot

show_versions
=============
.. autofunction:: show_versions

get_version
===========
.. autofunction:: get_version

VersionInfo
===========
.. autoclass:: VersionInfo
   :members:

LibraryInfo
===========
.. autoclass:: LibraryInfo
   :members:
