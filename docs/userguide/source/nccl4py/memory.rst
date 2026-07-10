.. py:currentmodule:: nccl.core

*****************
Memory Management
*****************

NCCL-backed device memory allocation; see :ref:`mem_allocator` for
usage details. For zero-copy registration of existing buffers, see
:py:meth:`Communicator.register_buffer` and
:py:meth:`Communicator.register_window`.

mem_alloc
---------
.. autofunction:: mem_alloc

mem_free
--------
.. autofunction:: mem_free
