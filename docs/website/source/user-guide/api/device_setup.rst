.. _device_api_setup:

Device API – Host-Side Setup
*****************************

Host-Side Setup
===============

**Host functions and types.** The following are for use in host code: creating and destroying device communicators,
querying properties, and the requirement and property types. The :c:type:`ncclDevComm` structure is then passed to
device code.

ncclTeam
--------

.. c:type:: ncclTeam_t

   Host-side typedef for ``struct ncclTeam``, describing a subset of communicator ranks.
   See :cpp:struct:`ncclTeam` for its fields and :ref:`devapi_teams` for the predefined teams.

ncclDevComm
-----------

.. c:type:: ncclDevComm

   A structure describing a device communicator, as created on the host side
   using :c:func:`ncclDevCommCreate`. The structure is used primarily on the
   device side. In general, fields in this struct are considered internal and
   should not be accessed by users. An exception is made for the following fields,
   which are guaranteed to be stable across NCCL versions:

   .. c:member:: int rank

      The rank within the communicator.

   .. c:member:: int nRanks

      The size of the communicator.

   .. c:member:: int lsaRank
   .. c:member:: int lsaSize

      Rank within the local :ref:`LSA <device_api_lsa>` team and its size (see :ref:`devapi_teams`).

   .. c:member:: uint32_t ginContextCount

      The number of supported GIN contexts (see :cpp:class:`ncclGin`; available
      since NCCL 2.28.7).

ncclDevCommCreate
-----------------

.. c:function:: ncclResult_t ncclDevCommCreate(ncclComm_t comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* outDevComm)

   Creates a new device communicator (see :c:type:`ncclDevComm`) corresponding to the supplied host-side communicator
   *comm*.  The result is returned in the *outDevComm* buffer (which needs to be supplied by the caller).  The caller needs
   to also provide a filled-in list of requirements via the *reqs* argument (see :c:type:`ncclDevCommRequirements`); the
   function will allocate any necessary resources to meet them. It is recommended to call :c:func:`ncclCommQueryProperties`
   before calling the function; the function will fail if the specified requirements are not supported. Since this is a
   collective call, every rank in the communicator needs to participate.  If called within a group, *outDevComm* may not be
   filled in until ``ncclGroupEnd()`` has completed.

   Note that this is a *host-side* function.

ncclDevCommDestroy
------------------

.. c:function:: ncclResult_t ncclDevCommDestroy(ncclComm_t comm, struct ncclDevComm const* devComm)

   Destroys a device communicator (see :c:type:`ncclDevComm`) previously created using :c:func:`ncclDevCommCreate` and
   releases any allocated resources.  The caller must ensure that no device kernel that uses this device communicator could
   be running at the time this function is invoked.

   Note that this is a *host-side* function.

ncclDevCommRequirements
-----------------------

.. c:type:: ncclDevCommRequirements

   A host-side structure specifying the list of requirements when creating device communicators (see
   :c:type:`ncclDevComm`). Since NCCL 2.29, this struct must be initialized using ``NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER``.

   .. c:member:: bool lsaMultimem

      Specifies whether multimem support is required for all :ref:`LSA <device_api_lsa>` ranks.

   .. c:member:: int lsaBarrierCount

      Specifies the number of memory barriers to allocate (see :cpp:class:`ncclLsaBarrierSession`). These barriers are
      necessary to write fused kernels and may be required by building blocks such as those in :ref:`device_api_reducecopy`.

   .. c:member:: int railGinBarrierCount

      Specifies the number of railed network barriers to allocate (see :cpp:class:`ncclGinBarrierSession`;
      available since NCCL 2.28.7).

   .. c:member:: int barrierCount

      Specifies the number of network/rail hybrid barriers to allocate (see :cpp:class:`ncclBarrierSession`; available since NCCL 2.28.7).

   .. c:member:: int ginContextCount

      Requested number of GIN contexts. The default is ``4``. This is a hint; after creation, use
      :c:member:`ncclDevComm.ginContextCount` to obtain the available count.

   .. c:member:: int ginQueueDepth

      Requested GIN queue depth. The default, ``0``, leaves selection to the GIN provider.
      Supported depths depend on the provider. When using ``ncclGinOptFlagsMaySkipCreditCheck``, callers
      must keep outstanding requests within the queue's available capacity.

   .. c:member:: bool ginExclusiveContexts

      Leave this option at its initializer default, ``false``.

   .. c:member:: int lsaLLA2ABlockCount
   .. c:member:: int lsaLLA2ASlotCount

      Leave these fields at their initializer defaults, ``0``. Request low-latency exchange resources
      with :c:func:`ncclLLA2ACreateRequirement` instead.

   .. c:member:: int ginSignalCount

      Specifies the number of network signals to allocate (see :cpp:type:`ncclGinSignal_t`; available since NCCL 2.28.7).

   .. c:member:: int ginCounterCount

      Specifies the number of network counters to allocate (see :cpp:type:`ncclGinCounter_t`; available since NCCL 2.28.7).

   .. c:member:: bool ginForceEnable

      **Deprecated since NCCL 2.29.7.** Forces GIN (GPU-Initiated Networking) support to be enabled by automatically setting
      ``ginConnectionType`` to :c:enumerator:`NCCL_GIN_CONNECTION_FULL <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_FULL>`. This field is deprecated in favor of explicitly
      setting :c:member:`ginConnectionType` to the desired value. When set to ``true``, it overrides the
      ``ginConnectionType`` field. New code should use :c:member:`ginConnectionType` directly instead of this field.
      Available since NCCL 2.28.7.

   .. c:member:: ncclGinConnectionType_t ginConnectionType

      Specifies the type of GIN connection to establish (see :c:enum:`ncclGinConnectionType_t`).
      If GIN resources are requested (e.g., ``ginSignalCount``, ``ginCounterCount``, ``barrierCount``, or
      ``railGinBarrierCount``) while this field is :c:enumerator:`NCCL_GIN_CONNECTION_NONE <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_NONE>`, device communicator
      creation fails with :c:enumerator:`ncclInvalidArgument <ncclResult_t.ncclInvalidArgument>`. Available since NCCL 2.29.7.

   .. c:member:: int ginCustomStride

      Specifies the rank stride when :c:member:`ginConnectionType` is :c:enumerator:`NCCL_GIN_CONNECTION_CUSTOM_STRIDE <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_CUSTOM_STRIDE>`.
      Available since NCCL 2.31.

   .. c:member:: int* ginCustomArray

      An array of length :c:member:`ginCustomArrayCount` which contains a list of peers to connect to.
      This array is only valid when :c:member:`ginConnectionType` is
      :c:enumerator:`NCCL_GIN_CONNECTION_CUSTOM_ARRAY <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_CUSTOM_ARRAY>`.
      Connection requests must be bi-directional: if rank *j* appears in rank *i*'s array, rank *i* must also appear
      in rank *j*'s array. Available since NCCL 2.32.4.

   .. c:member:: int ginCustomArrayCount

      Number of entries in :c:member:`ginCustomArray`. Defaults to ``0``. Available since NCCL 2.32.4.

   .. c:member:: ncclDevResourceRequirements_t* resourceRequirementsList

      Linked list of additional resources to allocate during :c:func:`ncclDevCommCreate`. Use ``NULL``
      when no additional resources are needed; see :ref:`device_api_resource_requirements`.

   .. c:member:: ncclTeamRequirements_t* teamRequirementsList

      Linked list of additional team requirements. Use ``NULL`` when no additional teams need resources;
      see :c:type:`ncclTeamRequirements_t`.

   .. c:member:: int ginTrafficClass

      Specifies the GIN traffic class. See :ref:`communicators_qos` for more details. Available since NCCL 2.30.3.

   .. c:member:: int worldGinBarrierCount

      Specifies the number of world network barriers to allocate. Available since NCCL 2.30.3.

   .. c:member:: bool ginStrongSignalsRequired

      Specifies whether GIN strong signals are required.
      Set to false if kernels using this communicator will not use strong signal operations
      (such as :cpp:struct:`ncclGin_StrongSignalInc` and :cpp:struct:`ncclGin_StrongVASignalAdd`).
      Default is true. Available since NCCL 2.30.5.


   .. c:member:: bool ginVaSignalsRequired

      Specifies whether GIN VA signals are required. Set to false if kernels using this communicator
      do not use GIN VA signals (such as :cpp:struct:`ncclGin_WeakVASignalInc` and :cpp:struct:`ncclGin_StrongVASignalAdd`).
      Default is true. Available since NCCL 2.30.5.

   .. c:member:: ncclGinType_t ginType

      Requests that the device communicator be connected using a specific GIN backend (see
      :c:enum:`ncclGinType_t`).
      :c:func:`ncclDevCommCreate` fails if set to a specific backend that is not supported.
      The default value is :c:enumerator:`NCCL_GIN_TYPE_NONE <ncclGinType_t.NCCL_GIN_TYPE_NONE>`,
      which automatically selects a backend that is supported by the communicator.

      Multiple GIN backends can be used at once by creating two device communicators, each with a different ginType.
      See :ref:`deviceapi_gin_multi_backend` for a worked example. Available since NCCL 2.31.

   .. c:member:: bool useRuntimeVersion

      If ``true``, :c:func:`ncclDevCommCreate` initializes a device
      communicator for device code built against the runtime NCCL version,
      e.g., JIT-compiled device code. Allocate the output buffer passed to
      :c:func:`ncclDevCommCreate` using
      ``ncclCommProperties_t.devCommRuntimeVersionSize`` bytes. If ``false``,
      the compile-time version is used. Available since
      NCCL 2.31.

   .. c:member:: int cftCaps

      Bitmask of CFT capabilities requested for the device communicator (see :ref:`device_api_cft`).
      Set :c:enumerator:`NCCL_CFT <ncclCftCap_t.NCCL_CFT>` for unicast CFT logical endpoint support,
      :c:enumerator:`NCCL_CFT_MULTIMEM <ncclCftCap_t.NCCL_CFT_MULTIMEM>` for multicast CFT support, or
      :c:enumerator:`NCCL_CFT_NONE <ncclCftCap_t.NCCL_CFT_NONE>` for no CFT support (default).
      Available since NCCL 2.31.

   .. c:member:: int cftBarrierCount

      Specifies the number of CFT barriers to allocate (see :cpp:class:`ncclCftBarrierSession`).
      Available since NCCL 2.31.

ncclCommQueryProperties
-----------------------

.. c:function:: ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props)

   Exposes communicator properties by filling in *props*. Before calling this function, *props* must be initialized
   using ``NCCL_COMM_PROPERTIES_INITIALIZER``. Introduced in NCCL 2.29.

   Note that this is a *host-side* function.

ncclCommProperties_t
--------------------

.. c:type:: ncclCommProperties_t

   A structure describing the properties of the communicator. Introduced in NCCL 2.29. Properties include:

   .. c:member:: int rank

      Rank within the communicator.

   .. c:member:: int nRanks

      Size of the communicator.

   .. c:member:: int cudaDev

      CUDA device index.

   .. c:member:: int nvmlDev

      NVML device index.

   .. c:member:: bool deviceApiSupport

      Whether the device API is supported. If false, a :c:type:`ncclDevComm` cannot be created.

   .. c:member:: bool multimemSupport

      Whether ranks in the same :ref:`LSA <device_api_lsa>` team can communicate using multimem. If false,
      a :c:type:`ncclDevComm` cannot be created with multimem resources.

   .. c:member:: ncclGinType_t ginType

      **Deprecated since NCCL 2.31.** The GIN type supported by the communicator. If equal to :c:enumerator:`NCCL_GIN_TYPE_NONE <ncclGinType_t.NCCL_GIN_TYPE_NONE>`, a
      :c:type:`ncclDevComm` cannot be created with GIN connection type :c:enumerator:`NCCL_GIN_CONNECTION_FULL <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_FULL>`.
      Available since NCCL 2.29.2.
      This field only reports a single GIN type and predates multi-backend GIN support. New code should use
      :c:member:`ginConnectionType` together with :c:member:`ginSupport` instead.

   .. c:member:: int nLsaTeams

      The number of :ref:`LSA <device_api_lsa>` teams across the entire communicator. Available since NCCL 2.29.7.

   .. c:member:: bool hostRmaSupport

      Whether host-initiated RMA is supported by the communicator.

   .. c:member:: ncclGinType_t railedGinType

      **Deprecated since NCCL 2.31.** The railed GIN type supported by the communicator. If equal to :c:enumerator:`NCCL_GIN_TYPE_NONE <ncclGinType_t.NCCL_GIN_TYPE_NONE>`, a
      :c:type:`ncclDevComm` cannot be created with GIN connection type :c:enumerator:`NCCL_GIN_CONNECTION_RAIL <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_RAIL>`.
      Available since NCCL 2.29.7.
      This field only reports a single GIN type and predates multi-backend GIN support. New code should use
      :c:member:`ginConnectionType` together with :c:member:`ginSupport` instead.

   .. c:member:: uint64_t commHash

      Communicator hash identifier shared across all ranks in the communicator. Available since NCCL 2.31.

   .. c:member:: int ginMinStride

      The minimum value allowed for :c:member:`ginCustomStride <ncclDevCommRequirements.ginCustomStride>`. This value is based
      on the network connectivity. For example, when :ref:`env_NCCL_CROSS_NIC` is ``0``, ``ginMinStride`` is increased so
      that strides spanning rail boundaries are not possible. ``ginMinStride`` is ``INT_MAX`` if GIN connectivity
      does not follow a uniform stride pattern, in which case :c:enumerator:`NCCL_GIN_CONNECTION_CUSTOM_STRIDE <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_CUSTOM_STRIDE>` cannot
      be used. Available since NCCL 2.31.

   .. c:member:: ncclGinConnectionType_t ginConnectionType

      The type of GIN connection established across the communicator (see :c:enum:`ncclGinConnectionType_t`).
      This reflects the connectivity actually negotiated for the communicator (e.g.,
      :c:enumerator:`NCCL_GIN_CONNECTION_NONE <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_NONE>` if GIN is unavailable,
      :c:enumerator:`NCCL_GIN_CONNECTION_FULL <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_FULL>`, or
      :c:enumerator:`NCCL_GIN_CONNECTION_RAIL <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_RAIL>`). Use this field together with
      :c:member:`ginSupport` to determine which GIN backends are available before setting
      :c:member:`ginConnectionType <ncclDevCommRequirements.ginConnectionType>` on :c:type:`ncclDevCommRequirements`.
      Available since NCCL 2.31.

   .. c:member:: bool ginSupport[64]

      Determines which GIN backends are supported.
      If ``ginSupport[gin_type]`` is ``true``, then ``gin_type`` is supported by every rank of the communicator.
      The value ``gin_type`` is one of the values from :c:enum:`ncclGinType_t`.
      See :ref:`deviceapi_gin_multi_backend`. Available since NCCL 2.31.

   .. c:member:: size_t devCommRuntimeVersionSize

      Number of bytes required for a device communicator targeting the runtime
      NCCL version. This can differ from ``sizeof(ncclDevComm_t)`` when the
      runtime NCCL version differs from the version used to compile the
      application. Available since NCCL 2.31.

   .. c:member:: bool cftSupport

      Whether the communicator supports unicast CFT operations. Available
      since NCCL 2.32.

   .. c:member:: bool cftMulticastSupport

      Whether the communicator supports multicast CFT operations. Available
      since NCCL 2.32.

   .. c:member:: bool cftCountedSupport

      Whether the communicator supports counted CFT operations. Available
      since NCCL 2.32.


ncclGinType_t
-------------

.. c:enum:: ncclGinType_t

   GIN type. Communication between different GIN types is not supported. Possible values include:

   .. c:enumerator:: NCCL_GIN_TYPE_NONE

      GIN is not supported.

   .. c:enumerator:: NCCL_GIN_TYPE_PROXY

      Host Proxy GIN type.

   .. c:enumerator:: NCCL_GIN_TYPE_GDAKI

      GPUDirect Async Kernel-Initiated (GDAKI) GIN type.

   .. c:enumerator:: NCCL_GIN_TYPE_GPI

      GPU-Push Interface (GPI) GIN type. Requires SpectrumX - see
      `SpectrumX documentation <https://networking-docs.nvidia.com/hpcxum/2.51/spectrum-x-nccl-plugin#GPU-Initiated-Networking-(GIN)>`_
      for details. Added as an experimental feature in NCCL 2.30.6.

   .. c:enumerator:: NCCL_GIN_TYPE_EFA_GDA

      AWS EFA GPUDirect Async (GDA) GIN type. Requires the AWS OFI plugin for NCCL -
      see AWS EFA `documentation <https://github.com/aws/aws-ofi-nccl>`_ and
      `getting started guide <https://github.com/aws/aws-ofi-nccl/blob/master/doc/gin-getting-started.md>`_
      for details. Available since NCCL 2.31.

.. _device_api_gin_connection_type:

ncclGinConnectionType_t
-----------------------

.. c:enum:: ncclGinConnectionType_t

   Specifies the type of GIN connection for device communicators. This enum controls whether GIN (GPU-Initiated
   Networking) resources should be allocated and what connection type to use. Used in :c:type:`ncclDevCommRequirements`
   when creating device communicators. Available since NCCL 2.29.7.

   .. c:enumerator:: NCCL_GIN_CONNECTION_NONE

      No GIN connectivity.

   .. c:enumerator:: NCCL_GIN_CONNECTION_FULL

      Full GIN connectivity. Each rank is connected to all other ranks.

   .. c:enumerator:: NCCL_GIN_CONNECTION_RAIL

      Railed GIN connectivity. Each rank is connected to other ranks in the same rail team.

   .. c:enumerator:: NCCL_GIN_CONNECTION_CUSTOM_STRIDE

      Strided GIN connectivity. Each rank is connected to other ranks separated by a fixed stride. Available since NCCL 2.31.

   .. c:enumerator:: NCCL_GIN_CONNECTION_CUSTOM_ARRAY

      Custom GIN connectivity. Each rank is connected to an arbitrary, user-specified set of peers. Available since NCCL 2.32.4.

.. _device_api_resource_requirements:

Additional Resource Requirements
=================================

Applications can request resources for custom teams or device communication sessions before calling
:c:func:`ncclDevCommCreate`. Link resource requirements through ``resourceRequirementsList`` and team requirements
through ``teamRequirementsList``. Keep the list entries and the objects referenced by their output pointers valid
until communicator creation has completed. Resource handles and signal/counter ranges are ready for use after
successful creation and remain valid until :c:func:`ncclDevCommDestroy`.

.. c:type:: ncclDevResourceRequirements_t

   A linked-list entry describing an additional allocation and GIN signals or counters.
   Initialize the entry to zero before setting its fields, or use one of the requirement helpers below.

   .. c:member:: ncclDevResourceRequirements_t* next

      Next entry, or ``NULL`` to terminate the list.

   .. c:member:: size_t bufferSize
   .. c:member:: size_t bufferAlign

      Allocation size and alignment in bytes.

   .. c:member:: ncclDevResourceHandle_t* outBufferHandle

      Optional output location for the allocation's resource handle.

   .. c:member:: int ginSignalCount
   .. c:member:: int ginCounterCount

      Additional indexed GIN signals and counters required by the caller.

   .. c:member:: ncclGinSignal_t* outGinSignalStart
   .. c:member:: ncclGinCounter_t* outGinCounterStart

      Optional output locations for the first index of each allocated range.

.. c:type:: ncclTeamRequirements_t

   A linked-list entry requesting resources for a team. Initialize the entry to zero before setting its fields.

   .. c:member:: ncclTeamRequirements_t* next

      Next entry, or ``NULL`` to terminate the list.

   .. c:member:: ncclTeam_t team

      Team for which resources are requested.

   .. c:member:: bool multimem

      Whether the team requires a multicast mapping.

   .. c:member:: ncclMultimemHandle_t* outMultimemHandle

      Optional output location for the multicast handle assigned during communicator creation.

Barrier requirements
---------------------

.. c:function:: ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq)

   Prepares *outReq* for *nBarriers* memory barriers over *team*. All team members must be load/store accessible.
   Link *outReq* into the resource-requirements list before creating the device communicator. Keep *outHandle*
   valid through creation, then pass it to an explicit-team :cpp:class:`ncclLsaBarrierSession` constructor.

.. c:function:: ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq)

   Prepares *outReq* for *nBarriers* GIN barriers over *team*. Request GIN connectivity covering every member
   of *team*. Link *outReq* into the resource-requirements list and keep *outHandle* valid through creation.
   The resulting handle can be passed to an explicit-team :cpp:class:`ncclGinBarrierSession` constructor.

These helpers initialize *outReq*, including its ``next`` field. Link entries after calling the helper.
All output pointers must be non-NULL, barrier counts must be non-negative, and the team layout must match
across participating ranks. These calls prepare requirements; they do not create a device communicator.

Low-latency exchange requirements
----------------------------------

.. c:type:: ncclLLA2AHandle_t

   A handle for resources used by :cpp:class:`ncclLLA2ASession`.

.. c:function:: int ncclLLA2ACalcSlots(int maxElts, int maxEltSize)

   Returns the slot count needed for up to *maxElts* elements, each at most *maxEltSize* bytes.
   Pass this result to :c:func:`ncclLLA2ACreateRequirement`. Include all element positions that the session will use.

.. c:function:: ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle, ncclDevResourceRequirements_t* outReq)

   Prepares resources for *nBlocks* independent low-latency sessions with *nSlots* slots per block.
   Both output pointers must be non-NULL. Link *outReq* into the resource-requirements list after this call
   and keep *outHandle* valid until device communicator creation completes.

.. _device_api_host_functions:

Host-Accessible Device Pointer Functions
=========================================

**Host functions.** The following are callable from host code only. They provide host-side access to device pointer
functionality, enabling host code to obtain pointers to :ref:`LSA <device_api_lsa>` memory regions.

All functions return ``ncclResult_t`` error codes. On success, ``ncclSuccess`` is returned.
On failure, appropriate error codes are returned (e.g., ``ncclInvalidArgument`` for invalid parameters,
``ncclInternalError`` for internal failures), unless otherwise specified.

The returned pointers are valid for the lifetime of the window.
Pointers should not be used after either the window or communicator is destroyed.
Obtained pointers are device pointers.

ncclGetLsaMultimemDevicePointer
--------------------------------

.. c:function:: ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr)

   Returns a multimem base pointer for the :ref:`LSA <device_api_lsa>` team associated with the given window.
   This function provides host-side access to the multimem memory functionality.

   *window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
   *outPtr* is the output parameter for the multimem pointer (must not be NULL).

   This function requires :ref:`LSA <device_api_lsa>` multimem support (multicast capability on the system). The window must be registered
   with a communicator that supports symmetric memory, and the hardware must support NVLink SHARP multicast functionality.

   .. note::
      If the system does not support multimem, the function returns ``ncclSuccess`` with ``*outPtr`` set to ``nullptr``.
      This allows applications to gracefully detect and handle the absence of multimem support without breaking
      the communicator. Users should check if the returned pointer is ``nullptr`` to determine availability.

   Example:

   .. code:: C

      void* multimemPtr;
      ncclResult_t result = ncclGetLsaMultimemDevicePointer(window, 0, &multimemPtr);
      if (result == ncclSuccess) {
          if (multimemPtr != nullptr) {
              // Use multimemPtr for multimem operations
          } else {
              // Multimem not supported, use fallback approach
          }
      }

ncclGetMultimemDevicePointer
----------------------------

.. c:function:: ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle_t multimem, void** outPtr)

   Returns a multimem base pointer using a provided multimem handle instead of the window's internal multimem.
   This function enables using external or custom multimem handles for pointer calculation.

   *window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
   *multimem* is the multimem handle containing the multimem base pointer (multimem.mcBasePtr must not be NULL).
   *outPtr* is the output parameter for the multimem pointer (must not be NULL).

   This function requires :ref:`LSA <device_api_lsa>` multimem support (multicast capability on the system).

   .. note::
      If the system does not support multimem, the function returns ``ncclSuccess`` with ``*outPtr`` set to ``nullptr``.
      The function validates that ``multimem.mcBasePtr`` is not nullptr before proceeding.

   Example:

   .. code:: C

      // Get multimem handle from device communicator setup
      ncclMultimemHandle_t customHandle;
      // ... (obtain handle)

      void* multimemPtr;
      ncclResult_t result = ncclGetMultimemDevicePointer(window, 0, customHandle, &multimemPtr);
      if (result == ncclSuccess) {
          if (multimemPtr != nullptr) {
              // Use multimemPtr for multimem operations with custom handle
          } else {
              // Multimem not supported, use fallback approach
          }
      }

ncclGetLsaDevicePointer
-----------------------

.. c:function:: ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr)

   Returns a load/store accessible pointer to the memory buffer of a specific :ref:`LSA <device_api_lsa>` peer
   within the window. This function provides host-side access to LSA pointer functionality using LSA rank directly.

   *window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window
   (must be >= 0 and < window size).
   *lsaRank* is the LSA rank of the target peer (must be >= 0 and < LSA team size).
   *outPtr* is the output parameter for the LSA pointer (must not be NULL).

   On success, ``ncclSuccess`` is returned and the LSA pointer is returned in ``outPtr``.


   The window must be registered with a communicator that supports LSA. The LSA rank must be within the valid range
   for the LSA team, and the target peer must be load/store accessible (P2P connectivity required).

   Example:

   .. code:: C

      void* lsaPtr;
      ncclResult_t result = ncclGetLsaDevicePointer(window, 0, 1, &lsaPtr);
      if (result == ncclSuccess) {
          // Use lsaPtr to access LSA peer 1's memory
      }

ncclGetPeerDevicePointer
------------------------

.. c:function:: ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr)

   Returns a load/store accessible pointer to the memory buffer of a specific world rank peer within the window.
   This function converts world rank to :ref:`LSA <device_api_lsa>` rank internally and provides host-side access
   to peer pointer functionality.

   *window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
   *peer* is the world rank of the target peer (must be >= 0 and < communicator size).
   *outPtr* is the output parameter for the peer pointer (must not be NULL).

   On success, ``ncclSuccess`` is returned and the peer pointer is returned in ``outPtr``.


   If the peer is not reachable via :ref:`LSA <device_api_lsa>` (not in LSA team), ``outPtr`` is set to NULL and
   ``ncclSuccess`` is returned.
   This matches the behavior of the device-side ``ncclGetPeerPointer`` function.

   The window must be registered with a communicator that supports LSA. The peer rank must be within the valid range
   for the communicator, and the target peer must be load/store accessible (P2P connectivity required).

   Example:

   .. code:: C

      void* peerPtr;
      ncclResult_t result = ncclGetPeerDevicePointer(window, 0, 2, &peerPtr);
      if (result == ncclSuccess) {
          if (peerPtr != NULL) {
              // Use peerPtr to access world rank 2's memory
          } else {
              // Peer 2 is not reachable via LSA
          }
      }
