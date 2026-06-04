.. _api_flags:

************************
NCCL API Supported Flags
************************

The following show all flags which are supported by NCCL APIs.

.. _win_flags:

Window Registration Flags
-------------------------

.. c:macro:: NCCL_WIN_DEFAULT

 Register buffer into NCCL window with default behavior. The default behavior allows users to
 pass any offset to the buffer head address as the input of NCCL collective operations. However,
 this behavior can cause suboptimal performance in NCCL due to the asymmetric buffer usage.

.. c:macro:: NCCL_WIN_COLL_SYMMETRIC

 Register buffer into NCCL window, and users need to guarantee the offset to the buffer head address
 from all ranks must be equal when calling NCCL collective operations. It allows NCCL to operate
 buffer in a symmetric way and provide the best performance.

.. c:macro:: NCCL_WIN_COLL_STRICT_ORDERING

  Register buffer into NCCL window while ensuring strict ordering for window operations using the IB Verbs transport.
  This flag is mostly intended for buffers used for GIN VA Signals (see :ref:`devapi_signals`).

.. _cta_policy_flags:

NCCL Communicator CTA Policy Flags
----------------------------------

.. c:macro:: NCCL_CTA_POLICY_DEFAULT

  Use the default CTA policy for NCCL communicator. In this policy, NCCL will automatically adjust resource usage and achieve
  maximal performance. This policy is suitable for most applications.

.. c:macro:: NCCL_CTA_POLICY_EFFICIENCY

  Use the CTA efficiency policy for NCCL communicator. In this policy, NCCL will optimize CTA usage and use minimal
  number of CTAs to achieve the decent performance when possible. This policy is suitable for applications which require
  better compute and communication overlap.

.. c:macro:: NCCL_CTA_POLICY_ZERO

  Use the Zero-CTA policy for NCCL communicator. In this policy, NCCL will use zero CTA whenever it can, even when that choice
  may sacrifice some performance. Select this mode when your application must preserve the maximum number of CTAs for compute kernels.

.. _comm_shrink_flags:

Communicator Shrink Flags
--------------------------

These flags modify the behavior of the ``ncclCommShrink`` operation.

.. c:macro:: NCCL_SHRINK_DEFAULT

   Default behavior. Shrink the parent communicator without affecting ongoing operations.
   Value: ``0x00``.

.. c:macro:: NCCL_SHRINK_ABORT

   First, terminate ongoing parent communicator operations, and then proceed with shrinking the communicator.
   This is used for error recovery scenarios where the parent communicator might be in a hung state.
   Resources of parent comm are still not freed, users should decide whether to call ncclCommAbort on the parent communicator after shrink.
   Value: ``0x01``.
