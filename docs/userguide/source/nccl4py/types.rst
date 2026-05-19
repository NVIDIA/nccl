.. py:currentmodule:: nccl.core

*******************
Types and Constants
*******************

Type wrappers, predefined value constants, and type aliases that appear
in public method signatures.

Data type
=========

NcclDataType
------------
.. autoclass:: NcclDataType
   :members:

Predefined data type constants
------------------------------

Module-level :py:class:`NcclDataType` instances for use as the ``dtype``
argument of buffer specs.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Constant
     - Maps to
   * - ``nccl.core.INT8``
     - :py:attr:`NcclDataType.INT8`
   * - ``nccl.core.CHAR``
     - :py:attr:`NcclDataType.CHAR`
   * - ``nccl.core.UINT8``
     - :py:attr:`NcclDataType.UINT8`
   * - ``nccl.core.INT32``
     - :py:attr:`NcclDataType.INT32`
   * - ``nccl.core.INT``
     - :py:attr:`NcclDataType.INT`
   * - ``nccl.core.UINT32``
     - :py:attr:`NcclDataType.UINT32`
   * - ``nccl.core.INT64``
     - :py:attr:`NcclDataType.INT64`
   * - ``nccl.core.UINT64``
     - :py:attr:`NcclDataType.UINT64`
   * - ``nccl.core.FLOAT16``
     - :py:attr:`NcclDataType.FLOAT16`
   * - ``nccl.core.HALF``
     - :py:attr:`NcclDataType.HALF`
   * - ``nccl.core.FLOAT32``
     - :py:attr:`NcclDataType.FLOAT32`
   * - ``nccl.core.FLOAT``
     - :py:attr:`NcclDataType.FLOAT`
   * - ``nccl.core.FLOAT64``
     - :py:attr:`NcclDataType.FLOAT64`
   * - ``nccl.core.DOUBLE``
     - :py:attr:`NcclDataType.DOUBLE`
   * - ``nccl.core.BFLOAT16``
     - :py:attr:`NcclDataType.BFLOAT16`
   * - ``nccl.core.FLOAT8E4M3``
     - :py:attr:`NcclDataType.FLOAT8E4M3`
   * - ``nccl.core.FLOAT8E5M2``
     - :py:attr:`NcclDataType.FLOAT8E5M2`

Reduction operator
==================

NcclRedOp
---------
.. autoclass:: NcclRedOp
   :members:

Predefined reduction operators
------------------------------

Module-level :py:class:`NcclRedOp` instances for use as the ``op`` argument
of reduction collectives. User-defined operators are created via
:py:meth:`Communicator.create_pre_mul_sum`.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Constant
     - Maps to
   * - ``nccl.core.SUM``
     - :py:attr:`NcclRedOp.SUM`
   * - ``nccl.core.PROD``
     - :py:attr:`NcclRedOp.PROD`
   * - ``nccl.core.MAX``
     - :py:attr:`NcclRedOp.MAX`
   * - ``nccl.core.MIN``
     - :py:attr:`NcclRedOp.MIN`
   * - ``nccl.core.AVG``
     - :py:attr:`NcclRedOp.AVG`

Exceptions
==========

NcclInvalid
-----------

Python-side validation exception, raised when a public API receives a
malformed argument before it reaches NCCL itself.

.. autoexception:: NcclInvalid
