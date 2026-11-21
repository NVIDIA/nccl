.. py:currentmodule:: nccl.core

********
Versions
********

NCCL4Py exposes helpers to inspect the installed NCCL stack: ``nccl4py``
itself, the version of the NCCL headers its bindings were generated from,
and the loaded ``libnccl.so``.

.. code-block:: python

    import nccl.core

    nccl.core.show_versions()      # human-readable block to stdout
    v = nccl.core.get_version()    # programmatic snapshot

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
