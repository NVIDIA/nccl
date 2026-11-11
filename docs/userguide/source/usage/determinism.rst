.. _determinism:

******************************
Determinism of NCCL Reductions
******************************

NCCL collective operations that perform reductions on input data (e.g., :ref:`AllReduce
<allreduce>`, :ref:`Reduce <reduce>`, and :ref:`ReduceScatter <reducescatter>`) might
produce different bitwise results in the output buffer, depending on the circumstances.
This is because floating point arithmetic is inherently non-associative; that is,
``a + (b + c)`` does not always equal ``(a + b) + c``.
For example, in FP32::

  ((2^30) + (-2^30)) + 1  ==  1
  (2^30) + ((-2^30) + 1)  ==  0

Any change in the order in which contributions are combined, or in the precision of the
intermediate accumulator, can therefore change the result bits of a floating-point
``ncclSum``, ``ncclProd``, ``ncclAvg``, or ``ncclPreMulSum``.
Such NCCL reductions are order-dependent, meaning the same logical reduction can return
different bits when the rank count, algorithm, protocol, message size, or collective
transport changes.
Integer reductions and ``ncclMin``/``ncclMax`` on ordinary finite values are not
order-sensitive in this way (special values such as NaN and signed zero are discussed
below).
There are several other factors that might affect NCCL reduction output, but a complete
list is beyond the scope of this documentation.
Instead, this page defines the common kinds of determinism, states which ones NCCL
guarantees wherever reduction arithmetic occurs, and offers guidance on how to obtain
deterministic results.

.. note::
   Repeatable NCCL collectives are necessary but not sufficient for end-to-end
   application determinism. Other sources of variation, such as non-deterministic
   CUDA kernels in the framework or atomics in user code, are outside NCCL's scope.

.. _determinism-kinds:

Kinds of determinism
--------------------

The computer science definition of a deterministic algorithm is one that always
produces the same output for a given input. For a floating-point NCCL
reduction, that definition hides several ambiguities. For example, the "input"
is more than the values themselves, because where they sit in the buffer and
how the communication is configured might affect the result bits. Likewise,
the "output" may mean one rank's result or every rank's result. The word
"determinism" therefore covers several different properties, and much confusion
comes from mixing them up.

When a NCCL reduction is called "deterministic", it may refer to any of at
least seven distinct properties. The properties answer different questions and require
different tests.

.. role:: guaranteed
.. role:: not-guaranteed

.. list-table::
   :header-rows: 1
   :widths: 22 48 30 14

   * - Property
     - Question answered
     - How to test it
     - NCCL guarantees
   * - Same-plan repeatability
     - Do identical input bits and an identical :ref:`effective execution plan
       <determinism-plan>` produce
       identical output bits on repeated executions, including across process restarts?
     - Compare output bytes against the first execution.
     - :guaranteed:`✓`
   * - Cross-rank identity
     - For an AllReduce, do all ranks receive exactly the same result bits in one
       execution?
     - Compare output bytes (or hashes) across ranks.
     - :guaranteed:`✓`
   * - Numerical accuracy
     - How close is the output to the true real-number result?
     - Compare against a gold-standard within some tolerance.
     - Application/tolerance dependent
   * - Order-independent result
     - Is the output the same no matter the order in which the inputs are combined?
     - Compare against an exact reference, for example a sum computed in
       arbitrary-precision arithmetic.
     - :not-guaranteed:`✗`
   * - Cross-buffer identity
     - If the same per-element inputs are placed at a different position in the
       buffers (a different element index, or a different address, offset, or
       alignment), are the output bits the same?
     - Compare output bytes for the same values at two different buffer positions.
     - :not-guaranteed:`✗`
   * - Plan-invariant reproducibility
     - Are result bits preserved when the algorithm, protocol, channel count, topology
       graph, or rank placement changes?
     - Compare output bytes across configurations.
     - :not-guaranteed:`✗`
   * - Cross-stack reproducibility
     - Are result bits preserved across NCCL, CUDA, compiler, driver, GPU, or fabric
       versions?
     - Compare output bytes across versions and hardware.
     - :not-guaranteed:`✗`

**NCCL guarantees same-plan repeatability and cross-rank identity.** Throughout this
page, "determinism" refers to the combination of these two guarantees unless another
property is named explicitly.

A few consequences of these definitions are worth stating:

* Numerical accuracy does not imply repeatability. A tolerance-based test can accept
  two different results.
* An order-independent result would imply every reproducibility property in the table,
  since there is only one correct answer regardless of order. NCCL's floating-point
  reductions do not provide it; "deterministic" does not mean "exact".
* Repeatability does not imply an order-independent result. It only says that the same
  numerical program produced the same bits.
* Cross-buffer identity fails because buffer position is part of the execution plan,
  not because NCCL reduces in a random order. In every NCCL algorithm that
  splits the buffer across channels, the element index selects the channel and the
  chunk within it, and different channels and chunks may combine ranks in a different
  order. The address and
  alignment can change which kernel is eligible, and in NVSwitch reductions
  they change which neighboring values an element is packed with, which affects
  its rounding. A given value at a given position is still repeatable.
* Plan-invariant and cross-stack reproducibility are much stronger properties. For
  floating-point ``ncclSum``, ``ncclProd``, ``ncclAvg``, and ``ncclPreMulSum`` they are
  generally incompatible with performance-driven changes to reduction order or
  accumulator precision, and NCCL does not promise them.

.. _determinism-background:

Background: how NCCL reduces data
---------------------------------

.. _determinism-plan:

The effective execution plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A call such as :c:func:`ncclAllReduce` does not run a single fixed program. NCCL first
lowers the reduction operation, decides whether a symmetric kernel or copy-engine method
is eligible, selects an algorithm (Ring, Tree, CollNet, NVLS, ...) and protocol (LL,
LL128, Simple) from its cost model, fixes the number of channels and the chunking, and
builds rings and trees from the detected topology and the rank-to-GPU mapping. The result
is the *effective execution plan*. Different plans combine the ranks' contributions in
different orders, so they are different numerical programs and are not expected to
produce identical floating-point bits.

"Identical setup" is often understood as "the same number of ranks on the same kind of
machines". That is not sufficient. All of the following are part of the plan, and
changing any of them can legitimately change floating-point result bits.

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Dimension
     - Examples
     - Why it matters
   * - Collective semantics
     - Function, count, datatype, reduction operation, root, ``ncclPreMulSum`` scalar
     - Defines inputs, the reduction primitive, and pre/post scaling
   * - Communicator identity
     - World size, rank-to-GPU mapping, node membership, local-rank order
     - Defines ring order, tree fan-in, and peer order
   * - Topology graph
     - NVLink/PCIe/NIC topology and detected link speeds, channel rings and trees,
       graph or topology files
     - Defines which ranks combine and in what hierarchy
   * - Implementation path
     - Algorithm and protocol, symmetric kernel, copy-engine method
     - Selects a distinct numerical implementation
   * - Launch geometry
     - Channels/CTAs, threads, chunk and slice sizes, aggregation
     - Maps elements onto channels, each with its own ring or tree
   * - Buffer state
     - In-place vs. out-of-place, user buffer registration, symmetric windows,
       alignment, memory type
     - Controls which kernels are eligible and which direct paths are used
   * - Submission context
     - Group calls, number of aggregated operations, CUDA Graph capture, CTA policy
     - Enters scheduling and tuning and can change the selected method
   * - Configuration
     - :ref:`NCCL_ALGO`, :ref:`NCCL_PROTO`, channel and CTA limits, communicator config
     - Forces or excludes execution plans
   * - Plugins
     - Tuner, network, and collective-network plugins and their versions
     - May override tuning decisions or perform the arithmetic externally
   * - Software stack
     - NCCL version and build, CUDA toolkit, driver
     - Controls instruction selection, accumulator type, and available paths
   * - Hardware stack
     - GPU architecture, NVLink/NVSwitch generation, NIC and fabric firmware
     - Controls available offloads and their implementation

NCCL's own selection logic is deterministic for fixed inputs: given the same inputs
listed above, NCCL builds the same plan. What NCCL cannot do is build the same plan when
the inputs change, for example when a job restarts with a different rank-to-GPU mapping.

Where the arithmetic happens
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depending on the plan, the reduction is performed in one of three places:

* **GPU software reduction** (Ring, Tree, PAT, symmetric kernels, and the GPU stages of
  CollNet and NVLS). Operands are combined in a fixed order determined by the plan.
* **NVSwitch** (NVLink SHARP, via ``multimem`` instructions). The order is defined by the
  hardware. See :ref:`determinism-nvls`.
* **InfiniBand fabric** (CollNet with the SHARP plugin). The order is defined by SHARP.
  See :ref:`determinism-collnet`.

Some paths accumulate in a wider type than the input, for example FP32 for FP16 or BF16
sums. Widening reduces sensitivity to reduction order but does not eliminate it.
Repeatability comes from a fixed reduction order and fixed rounding behavior, not from a
wider accumulator.

.. _determinism-guarantee:

What NCCL guarantees: same-plan repeatability
---------------------------------------------

For a valid collective invocation with identical input bits, NCCL produces identical
output bits, per output element, whenever the :ref:`effective execution plan
<determinism-plan>` is unchanged. This holds across repeated calls on the same
communicator, across communicator re-creation, and across process restarts.

In particular:

* Software reductions combine operands in a fixed order for a fixed plan and are bitwise
  repeatable by construction.
* NVLink SHARP (NVLS) reductions are bitwise repeatable on the software stack described
  in :ref:`determinism-nvls`.
* InfiniBand SHARP reductions are bitwise repeatable for an identical SHARP library,
  firmware, and fabric configuration.
* Operations that only move data (AllGather, Broadcast, Gather, Scatter, AlltoAll,
  Send/Recv, and the copy-engine collectives) produce identical output bytes, because
  each output byte has exactly one defined source.
* All ranks of an AllReduce receive the same result bits for a given element in a given
  execution (cross-rank identity).

NCCL does **not** promise identical floating-point result bits:

* across different algorithms, protocols, symmetric kernels, or channel counts;
* across different rank-to-GPU mappings or topology graphs, even with the same number of
  ranks;
* across different NCCL versions, CUDA toolkits, drivers, GPU or NVSwitch generations,
  or network/fabric stacks;
* for in-network reductions performed by a collective-network provider other than
  NVIDIA SHARP, unless that provider documents an equivalent guarantee;
* for user-authored kernels built on the :ref:`Device API <device_api_lsa>` (see
  :ref:`determinism-deviceapi`).

Any of the above may change reduction order or accumulator precision while remaining a
valid floating-point reduction.

The guarantee also requires a valid invocation: matching collectives on every rank,
identical input bits, no modification of input or in-place buffers before NCCL has
finished with them, no unsynchronized writers racing with NCCL, and stream, CUDA Graph,
and communicator dependencies that establish the intended order (see
:ref:`multi-thread-concurrent-usage` and :ref:`NCCL_LAUNCH_ORDER_IMPLICIT`). Mismatched
collectives and buffer races are undefined behavior and are outside the guarantee.

.. _determinism-nvls:

NVLink SHARP (NVLS) reductions
------------------------------

NVLink SHARP offloads reductions to the NVSwitch using ``multimem`` instructions (see
:ref:`env_NCCL_NVLS_ENABLE`). The NVSwitch combines contributions in a deterministic
order and rounds deterministically, so NVLS reductions are bitwise repeatable for an
identical plan.

On **Hopper**, this requires the following minimum kernel driver and Fabric Manager
versions. Earlier Hopper software could program the NVSwitch with a different reduction
order depending on the order in which GPUs joined the multicast group, so the same job
could produce different bits after a restart.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Component
     - Minimum version
   * - NVIDIA kernel driver and Fabric Manager, 550 branch
     - 550.144.03
   * - NVIDIA kernel driver and Fabric Manager, 570 branch and later
     - 570.124.06
   * - NVIDIA kernel driver and Fabric Manager, 560 branch
     - No fix available; use 550.144.03 or later, or 570.124.06 or later

The fix is in the kernel driver and Fabric Manager. The NCCL version and the CUDA
toolkit and user-space library (``libcuda.so``) versions do not affect it.

**Blackwell and later** have no such driver or Fabric Manager requirement.

As with any other change to the hardware stack, bits are not expected to match across
GPU or NVSwitch generations.

.. note::

   **Known issue.** FP16 and BF16 NVLS AllReduce and ReduceScatter with a count that is
   not a multiple of 4 may return different bits for the last one to three elements
   between runs on identical input. Pad counts to a multiple of 4 or set
   :ref:`NCCL_NVLS_ENABLE=0 <env_NCCL_NVLS_ENABLE>` when bitwise repeatability is
   required.

.. _determinism-collnet:

In-network reductions (CollNet and InfiniBand SHARP)
----------------------------------------------------

With the CollNetDirect and CollNetChain algorithms, the inter-node arithmetic may be
performed by a collective-network plugin. NVIDIA InfiniBand SHARP uses a deterministic
reduction order and is bitwise repeatable for an identical SHARP library, firmware,
fabric configuration, and plan. Other
collective-network providers are covered only if they document an equivalent guarantee.
The GPU-side stages of CollNet collectives are ordinary software reductions.

.. _determinism-deviceapi:

Device API and user-authored kernels
------------------------------------

NCCL's collective guarantee does not extend to kernels composed from the :ref:`Device API
<device_api_lsa>`. A user-written kernel controls the peer traversal order, whether
contributions are consumed in a fixed or arrival order, the accumulator type, and
synchronization. A kernel that iterates peers in a fixed order with correct
synchronization is repeatable in the same way as NCCL's own kernels; one that consumes
contributions in arrival order or uses floating-point atomics is not. NCCL makes no
statement about either.

.. _determinism-special-values:

Special values and reduction operations
---------------------------------------

NCCL does not inspect or special-case NaN, infinities, or signed zeros. Each reduction
step applies the native arithmetic of the unit performing it: the CUDA math functions in
NCCL's GPU kernels, the NVSwitch reduction engine for NVLS, or the NIC and switch
hardware for CollNet/SHARP. The following therefore holds for NCCL's GPU kernels; the
hardware-offloaded paths follow their own arithmetic, which NCCL does not alter.

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Operation / type
     - Order sensitivity
     - Behavior with special values (GPU kernels)
   * - Integer ``ncclSum``, ``ncclProd``, ``ncclMin``, ``ncclMax``, ``ncclAvg``
     - None
     - Result value is order-independent.
   * - Floating ``ncclSum``, ``ncclProd``, ``ncclAvg``, ``ncclPreMulSum``
     - High
     - Non-associative; order and accumulator precision both matter. A NaN in any input
       propagates to the output, as does a NaN produced along the way (for example
       ``inf + -inf``). The payload and sign of the resulting NaN are not specified.
   * - Floating ``ncclMin``, ``ncclMax`` on finite values
     - None
     - The result is the extreme element itself, so it is order-independent; the only
       exception is a tie between ``-0.0`` and ``+0.0`` (see the signed zero row below).
   * - Floating ``ncclMin``, ``ncclMax`` with NaN
     - None
     - NaN is treated as missing data: when one operand is NaN the other operand is
       returned, matching C ``fmin``/``fmax``. The result is NaN only when every input
       element is NaN. Hardware-offloaded paths may differ.
   * - Floating ``ncclMin``, ``ncclMax`` with signed zero
     - Low
     - FP16 and BF16 kernels order ``-0.0 < +0.0``; for other types the sign of a zero
       result is not specified.
   * - FP8, FP16, BF16 ``ncclSum``, ``ncclProd``, ``ncclAvg``, ``ncclPreMulSum``
     - High
     - As for other floating sums, but a change of order flips result bits more often
       because these formats have few mantissa bits. Accumulator precision depends on
       the selected implementation, datatype, and reduction operation; both software and
       hardware-offloaded reductions may accumulate in a wider type than the input.

Within a fixed plan, all of these are repeatable, including NaN payloads and the sign of
zero. Across plans, only the integer and finite min/max results are expected to be
identical. Applications that need "any NaN anywhere makes the result NaN" semantics for
``ncclMin``/``ncclMax`` should check their inputs or outputs themselves rather than rely
on the reduction to surface it.

.. _determinism-practice:

Practical guidance
------------------

Keeping the plan fixed
^^^^^^^^^^^^^^^^^^^^^^

To obtain bitwise-repeatable results across runs and restarts:

* Use the same NCCL version and build, CUDA toolkit, and driver.
* Run on the same hardware with the same NVLink/NVSwitch and network configuration and
  firmware.
* Pin each rank to the same physical GPU and node across runs.
* Keep the same communicator membership, rank order, and local-rank order.
* Keep NCCL configuration identical: environment variables such as :ref:`NCCL_ALGO`,
  :ref:`NCCL_PROTO`, :ref:`NCCL_MIN_CTAS`/:ref:`NCCL_MAX_CTAS`, and
  :ref:`env_NCCL_NVLS_ENABLE`, communicator configuration (:ref:`ncclconfig`), and any
  topology or graph files.
* Use the same tuner, network, and collective-network plugins and versions.
* Issue the same collectives with the same buffer properties: in-place or out-of-place,
  registered or unregistered (see :ref:`user_buffer_reg`), the same alignment, the same
  grouping, and the same CUDA Graph usage.

Forcing a specific algorithm and protocol with ``NCCL_ALGO`` and ``NCCL_PROTO`` removes
one source of plan variation but is not required for repeatability, since NCCL's
automatic selection is itself deterministic for fixed inputs. It can, however, make it
easier to compare runs across slightly different environments, and it might cost
performance.

Diagnosing a suspected determinism problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most reports of "non-deterministic NCCL" turn out to be one of the following:

* **The plan changed.** A restart mapped ranks to different GPUs or nodes, a node was
  replaced, a message size crossed a tuning threshold, or a buffer became registered. Run
  with ``NCCL_DEBUG=INFO`` and ``NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,NVLS`` (see
  :ref:`NCCL_DEBUG_SUBSYS`) in both runs and compare the selected algorithm, protocol,
  channel count, rings and trees, and rank-to-GPU mapping before concluding that the
  kernel is non-deterministic.
* **An application race.** Input buffers were modified before NCCL finished reading them,
  or the output was read before the stream completed.
* **NVLS on Hopper with a driver or Fabric Manager older than required** (see
  :ref:`determinism-nvls`).
* **A non-NCCL source of variation**, such as non-deterministic framework kernels. Test
  by comparing the NCCL output buffers directly rather than downstream loss values.

If the plan is identical, the invocation is valid, and the NCCL output buffers still
differ byte for byte, that is a bug. Please report it with the ``NCCL_DEBUG=INFO`` output
from both runs.

Frequently asked questions
--------------------------

**Is NCCL deterministic?**
  NCCL guarantees same-plan repeatability: the same collective, with the same input bits
  and the same effective execution plan on the same software and hardware stack, produces
  the same output bits, including across process restarts. It does not guarantee the same
  bits across different plans, versions, or hardware.

**I ran the same training job twice on the same cluster and got different results. Is NCCL the cause?**
  Not necessarily. First check that the rank-to-GPU mapping, the selected algorithm and
  protocol, and the channel count were identical in both runs (see
  :ref:`determinism-practice`). On Hopper, also check the driver and Fabric Manager
  versions (see :ref:`determinism-nvls`). If the plan is identical and the NCCL output
  bits still differ, that is a bug; please report it.

**Does NVLink SHARP make results non-deterministic?**
  No, apart from the known issue noted in :ref:`determinism-nvls`. NVSwitch reduction
  order and rounding are deterministic by design. Hopper systems
  require the driver and Fabric Manager versions listed in :ref:`determinism-nvls`;
  Blackwell and later have no such requirement.

**Will Ring and Tree give me the same bits?**
  Not in general. They are different reduction orders. Both are individually repeatable.

**Do I need to set NCCL_ALGO and NCCL_PROTO to be deterministic?**
  No. NCCL's selection is deterministic for fixed inputs. Forcing them only helps when you
  cannot otherwise hold all tuning inputs constant between runs, and it might cost
  performance.

**Can I get the same bits after upgrading NCCL or the driver?**
  NCCL does not promise this. Upgrades may change reduction order or accumulator precision
  while remaining functionally correct. Treat cross-version comparison as a separate
  qualification step rather than an expectation.
