/*
* Copyright 2009-2020  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#include "nvToolsExt.h"

#ifndef NVTOOLSEXTV3_MEM_V1
#define NVTOOLSEXTV3_MEM_V1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define NVTX_EXT_MODULEID_MEM 1

/* \cond SHOW_HIDDEN
 * \brief A compatibility ID value used in structures and initialization to
 * identify version differences.
 */
#define NVTX_EXT_COMPATID_MEM 0x0102

/* \cond SHOW_HIDDEN
 * \brief This value is returned by functions that return `nvtxMemHeapHandle_t`,
 * if a tool is not attached.
 */
#define NVTX_MEM_HEAP_HANDLE_NO_TOOL ((nvtxMemHeapHandle_t)(intptr_t)-1)

/* \cond SHOW_HIDDEN
 * \brief This value is returned by functions that return `nvtxMemRegionHandle_t`
 * if a tool is not attached.
 */
#define NVTX_MEM_REGION_HANDLE_NO_TOOL ((nvtxMemRegionHandle_t)(intptr_t)-1)

/* \cond SHOW_HIDDEN
 * \brief This value is returned by functions that return `nvtxMemPermissionsHandle_t`
 * if a tool is not attached.
 */
#define NVTX_MEM_PERMISSIONS_HANDLE_NO_TOOL ((nvtxMemPermissionsHandle_t)-1)


/* \cond SHOW_HIDDEN
 * \brief This should not be used and is considered an error but defined to
 * detect an accidental use of zero or NULL.
 */
#define NVTX_MEM_HEAP_USAGE_UNKNOWN 0x0


/* \cond SHOW_HIDDEN
 * \brief This should not be used and is considered an error but defined to
 * detect an accidental use of zero or NULL.
 */
#define NVTX_MEM_TYPE_UNKNOWN 0x0


/*  ------------------------------------------------------------------------- */
/** \defgroup MEMORY Memory
 * See page \ref PAGE_MEMORY.
 * @{
 */

/**
 * \brief To indicate the full process virtual address space as a heap for
 * functions where a nvtxMemHeapHandle_t is accepted.
 *
 * The heap by default is always read-write-execute permissions without creating regions.
 * Regions created in this heap have read-write access by default but not execute.
 */
#define NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE ((nvtxMemHeapHandle_t)0)

/** \brief This heap is a sub-allocator.
 *
 * Heap created with this usage should not be accessed by the user until regions are registered.
 * Regions from a heap with this usage have read-write access by default but not execute.
 */
#define NVTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR 0x1

/**
 * \brief This is a heap of memory that has an explicit layout.
 *
 * The layout could be static or dynamic (calculated). This often represents an algorithm's
 * structures that are packed together. By default this heap is assumed to be accessible for
 * scopes where the memory is naturally accessible by hardware. Regions may be use to further
 * annotate or restrict access. A tool may have an option to be more strict, but special
 * consideration must be made for `NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE`.
 *
 * The behavior of this usage is similar to NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE but
 * a tool can use it to track special behaviors and reservation.
 *
 * Memory in a heap with this usage has read-write permissions by default but not execute without
 * creating regions. Regions created in this heap have the same default permission access.
 */
#define NVTX_MEM_HEAP_USAGE_TYPE_LAYOUT 0x2


/**
 * \brief Standard process userspace virtual addresses for linear allocations.
 *
 * APIs that map into this space, such as CUDA UVA should use this type.
 *
 * Relevant functions: cudaMalloc, cudaMallocManaged, cudaHostAlloc, cudaMallocHost
 * NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE is supported
 *
 * nvtxMemHeapRegister receives a heapDesc of type nvtxMemVirtualRangeDesc_t
 */
#define NVTX_MEM_TYPE_VIRTUAL_ADDRESS 0x1


/**
 * \brief To indicate you are modifying permissions to the process-wide
 * full virtual address space.
 *
 * This is a companion object to `NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE`.
 */
#define NVTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE ((nvtxMemPermissionsHandle_t)0)

#define NVTX_MEM_PERMISSIONS_CREATE_FLAGS_NONE 0x0
#define NVTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_READ 0x1
#define NVTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_WRITE 0x2
#define NVTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_ATOMIC 0x4


/* \cond SHOW_HIDDEN
 * \brief Forward declaration of opaque memory heap structure.
 */
struct nvtxMemHeap_v1;
typedef struct nvtxMemHeap_v1 nvtxMemHeap_t;

/** \brief A handle returned by a tool to represent a memory heap. */
typedef nvtxMemHeap_t* nvtxMemHeapHandle_t;

/* \cond SHOW_HIDDEN
 * \brief Forward declaration of opaque memory heap structure.
 */
struct nvtxMemRegion_v1;
typedef struct nvtxMemRegion_v1 nvtxMemRegion_t;

/** \brief A handle returned by a tool to represent a memory region. */
typedef nvtxMemRegion_t* nvtxMemRegionHandle_t;

/** \brief A reference to a memory region (by pointer or handle).
 * Which member of the union will be determined by a type or flag field outside.
 */
typedef union nvtxMemRegionRef_t
{
    void const* pointer;
    nvtxMemRegionHandle_t handle;
} nvtxMemRegionRef_t;

/* \cond SHOW_HIDDEN
 * \brief Forward declaration of opaque memory permissions structure
 */
struct nvtxMemPermissions_v1;
typedef struct nvtxMemPermissions_v1 nvtxMemPermissions_t;

/** \brief A handle returned by a tool to represent a memory permissions mask. */
typedef nvtxMemPermissions_t* nvtxMemPermissionsHandle_t;


typedef struct nvtxMemVirtualRangeDesc_v1
{
    size_t  size;
    void const*  ptr;
} nvtxMemVirtualRangeDesc_v1 ;
typedef nvtxMemVirtualRangeDesc_v1 nvtxMemVirtualRangeDesc_t;


/** \brief structure to describe a heap in process virtual memory. */
typedef struct nvtxMemHeapDesc_v1
{
    uint16_t extCompatID; /* Set to NVTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */
    uint32_t reserved0;

    /** \brief Usage characteristics of the heap
     *
     * Usage characteristics help tools like memcheckers, santiizer,
     * as well as other debugging and profiling tools to determine some
     * special behaviors they should apply to the heap and it's regions.
     * The value follows the convention NVTX_MEM_HEAP_USAGE_*
     *
     * Default Value is 0, which is invalid.
     */
    uint32_t usage;

    /** \brief Memory type characteristics of the heap
     *
     * The 'type' indicates how to interpret the ptr field of the heapDesc.
     * This is intended to support many additional types of memory, beyond
     * standard process virtual memory, such as API specific memory only
     * addressed by handles or multi-dimensional memory requiring more complex
     * descriptions to handle features like strides, tiling, or interlace.
     *
     * The values conforms to NVTX_MEM_TYPE_*
     *
     * The value in the field 'type' identifies the descriptor type that will
     * be in the field 'typeSpecificDesc'.  'typeSpecificDesc' is void* because
     * it is extensible.  Example usage is if type is NVTX_MEM_TYPE_VIRTUAL_ADDRESS,
     * then typeSpecificDesc points to a nvtxMemVirtualRangeDesc_t.
     *
     * Default Value is 0, which is invalid.
     */
    uint32_t type;

    /** \brief size of the heap memory descriptor pointed to by typeSpecificDesc
     *
     * Default Value is 0 which is invalid.
     */
    size_t typeSpecificDescSize;

    /** \brief Pointer to the heap memory descriptor
     *
     * The value in the field 'type' identifies the descriptor type that will
     * be in the field 'typeSpecificDesc'.  'typeSpecificDesc' is void* because
     * it is extensible.  Example usage is if type is NVTX_MEM_TYPE_VIRTUAL_ADDRESS,
     * then typeSpecificDesc points to a nvtxMemVirtualRangeDesc_t.
     *
     * Default Value is 0, which is invalid.
     */
    void const* typeSpecificDesc;

    /** \brief ID of the category the event is assigned to.
     *
     * A category is a user-controlled ID that can be used to group
     * events.  The tool may use category IDs to improve filtering or
     * enable grouping of events in the same category. The functions
     * \ref ::nvtxNameCategoryA or \ref ::nvtxNameCategoryW can be used
     * to name a category.
     *
     * Default Value is 0.
     */
    uint32_t category;

    /** \brief Message type specified in this attribute structure.
     *
     * Defines the message format of the attribute structure's \ref MESSAGE_FIELD
     * "message" field.
     *
     * Default Value is `NVTX_MESSAGE_UNKNOWN`.
     */
    uint32_t messageType;            /* nvtxMessageType_t */

    /** \brief Message assigned to this attribute structure. \anchor MESSAGE_FIELD
     *
     * The text message that is attached to an event.
     */
    nvtxMessageValue_t message;

} nvtxMemHeapDesc_v1 ;
typedef nvtxMemHeapDesc_v1 nvtxMemHeapDesc_t;

/**
 * \brief Create a memory heap to represent a object or range of memory that will be further
 * sub-divided into regions.
 *
 * The handle used to addrss the heap will depend on the heap's type.  Where the heap is virtual
 * memory accessible, the addrss of the heap's memory itself is it's handle. This will likewise
 * be returned from the function.
 *
 * For more advanced types, where the heap is not virtual memory accessible the tools may be
 * responsible for returning a void const * that that uniquely identifies the object. Please see
 * the description of each heap type for more details on whether this is expected to be a uniquely
 * generated by the tool or otherwise.
 */
NVTX_DECLSPEC nvtxMemHeapHandle_t NVTX_API nvtxMemHeapRegister(
    nvtxDomainHandle_t domain,
    nvtxMemHeapDesc_t const* desc);

 /** \brief Destroy a memory heap. */
NVTX_DECLSPEC void NVTX_API nvtxMemHeapUnregister(
    nvtxDomainHandle_t domain,
    nvtxMemHeapHandle_t heap);/* NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE is not supported */

/**
 * \brief Reset the memory heap wipes out any changes, as if it were a fresh heap.
 *
 * This includes invalidating all regions and their handles.
 */
NVTX_DECLSPEC void NVTX_API nvtxMemHeapReset(
    nvtxDomainHandle_t domain,
    nvtxMemHeapHandle_t heap); /* NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE is supported */

/**
 * \brief Register a region of memory inside of a heap.
 *
 * The heap refers the the heap within which the region resides. This can be from
 * `nvtxMemHeapRegister`, `NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE`, or one provided
 * from other extension API.
 *
 * The regionType arg will define which type is used in regionDescArray.
 * The most commonly used type is `NVTX_MEM_TYPE_VIRTUAL_ADDRESS`.
 * In this case regionDescElements is an array of `nvtxMemVirtualRangeDesc_t`.
 *
 * The regionCount arg is how many element are in regionDescArray and regionHandleArrayOut.
 *
 * The regionHandleArrayOut arg points to an array where the tool will provide region handles. If
 * a pointer is provided, it is expected to have regionCount elements. This pointer can be NULL if
 * regionType is NVTX_MEM_TYPE_VIRTUAL_ADDRESS. In this case, the user can use the pointer to the
 * virtual memory to reference the region in other related functions which accept nvtMemRegionRef_t.
 */
typedef struct nvtxMemRegionsRegisterBatch_v1
{
    uint16_t extCompatID; /* Set to NVTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t regionType; /* NVTX_MEM_TYPE_* */

    nvtxMemHeapHandle_t heap;

    size_t regionCount;
    size_t regionDescElementSize;
    void const* regionDescElements; /* This will also become the handle for this region. */
    nvtxMemRegionHandle_t* regionHandleElementsOut; /* This will also become the handle for this region. */

} nvtxMemRegionsRegisterBatch_v1;
typedef nvtxMemRegionsRegisterBatch_v1 nvtxMemRegionsRegisterBatch_t;

 /** \brief Register a region of memory inside of a heap of linear process virtual memory
 */
NVTX_DECLSPEC void NVTX_API nvtxMemRegionsRegister(
    nvtxDomainHandle_t domain,
    nvtxMemRegionsRegisterBatch_t const* desc);



/**
 * \brief Register a region of memory inside of a heap.
 *
 * The heap refers the the heap within which the region resides.
 * This can be from nvtxMemHeapRegister, NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE, or
 * one provided from other extension API.
 *
 * The regionType arg will define which type is used in regionDescArray.
 * The most commonly used type is NVTX_MEM_TYPE_VIRTUAL_ADDRESS.
 *
 * The regionCount arg is how many element are in regionDescArray and regionHandleArrayOut.
 *
 * The regionHandleArrayOut arg points to an array where the tool will provide region handles. If
 * a pointer if provided, it is expected to have regionCount elements. This pointer can be NULL if
 * regionType is NVTX_MEM_TYPE_VIRTUAL_ADDRESS. In this case, the user can use the pointer to the
 * virtual memory to reference the region in other related functions which accept nvtMemRegionRef_t.
 */
typedef struct nvtxMemRegionsResizeBatch_v1
{
    uint16_t extCompatID; /* Set to NVTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t regionType; /* NVTX_MEM_TYPE_* */

    size_t regionDescCount;
    size_t regionDescElementSize;
    void const* regionDescElements; /* This will also become the handle for this region. */

} nvtxMemRegionsResizeBatch_v1;
typedef nvtxMemRegionsResizeBatch_v1 nvtxMemRegionsResizeBatch_t;

 /** \brief Register a region of memory inside of a heap of linear process virtual memory
 */
NVTX_DECLSPEC void NVTX_API nvtxMemRegionsResize(
    nvtxDomainHandle_t domain,
    nvtxMemRegionsResizeBatch_t const* desc);


#define NVTX_MEM_REGION_REF_TYPE_UNKNOWN 0x0
#define NVTX_MEM_REGION_REF_TYPE_POINTER 0x1
#define NVTX_MEM_REGION_REF_TYPE_HANDLE 0x2

/**
 * \brief Register a region of memory inside of a heap.
 *
 * The heap refers the the heap within which the region resides.
 * This can be from nvtxMemHeapRegister, `NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE`, or
 * one provided from other extension API.
 *
 * The regionType arg will define which type is used in `regionDescArray`.
 * The most commonly used type is NVTX_MEM_TYPE_VIRTUAL_ADDRESS.
 *
 * The regionCount arg is how many element are in regionDescArray and regionHandleArrayOut.
 *
 * The regionHandleArrayOut arg points to an array where the tool will provide region handles.
 * If a pointer if provided, it is expected to have regionCount elements.
 * This pointer can be NULL if regionType is NVTX_MEM_TYPE_VIRTUAL_ADDRESS.  In this case,
 * the user can use the pointer to the virtual memory to reference the region in other
 * related functions which accept a nvtMemRegionRef_t.
 */
typedef struct nvtxMemRegionsUnregisterBatch_v1
{
    uint16_t extCompatID; /* Set to NVTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t refType; /* NVTX_MEM_REGION_REF_TYPE_* */

    size_t refCount; /* count of elements in refArray */
    size_t refElementSize;
    nvtxMemRegionRef_t const* refElements; /* This will also become the handle for this region. */

} nvtxMemRegionsUnregisterBatch_v1;
typedef nvtxMemRegionsUnregisterBatch_v1 nvtxMemRegionsUnregisterBatch_t;

/**
 * \brief Unregistration for regions of process virtual memory
 *
 * This is not necessary if the nvtx heap destroy function has been called that
 * contains this object.
 */
NVTX_DECLSPEC void NVTX_API nvtxMemRegionsUnregister(
    nvtxDomainHandle_t domain,
    nvtxMemRegionsUnregisterBatch_t const* desc);

typedef struct nvtxMemRegionNameDesc_v1
{
    uint32_t regionRefType; /* NVTX_MEM_REGION_REF_TYPE_* */
    uint32_t nameType; /* nvtxMessageType_t */

    nvtxMemRegionRef_t region;
    nvtxMessageValue_t name;

    uint32_t category;
    uint32_t reserved0;
} nvtxMemRegionNameDesc_v1;
typedef nvtxMemRegionNameDesc_v1 nvtxMemRegionNameDesc_t;


typedef struct nvtxMemRegionsNameBatch_v1
{
    uint16_t extCompatID; /* Set to NVTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t reserved0;

    size_t regionCount;
    size_t regionElementSize;
    nvtxMemRegionNameDesc_t const* regionElements;
    size_t reserved1;
} nvtxMemRegionsNameBatch_v1 ;
typedef nvtxMemRegionsNameBatch_v1 nvtxMemRegionsNameBatch_t;


 /** \brief Name or rename a region. */
NVTX_DECLSPEC void NVTX_API nvtxMemRegionsName(
    nvtxDomainHandle_t domain,
    nvtxMemRegionsNameBatch_t const* desc);

/** \brief There are no permissions for this memory. */
#define NVTX_MEM_PERMISSIONS_REGION_FLAGS_NONE 0x0

/** \brief The memory is readable. */
#define NVTX_MEM_PERMISSIONS_REGION_FLAGS_READ 0x1

/** \brief The memory is writable. */
#define NVTX_MEM_PERMISSIONS_REGION_FLAGS_WRITE 0x2

/** \brief The memory is for atomic RW. */
#define NVTX_MEM_PERMISSIONS_REGION_FLAGS_ATOMIC 0x4

/**
 * \brief The memory access permissions are reset for a region.
 *
 * This is as if never set, rather than documented defaults.  As as result any flags
 * indicating how unspecified regions are handle will affect this area.
 *
 * This should not be used with READ, WRITE, nor ATOMIC, as those flags would have no effect.
 */
#define NVTX_MEM_PERMISSIONS_REGION_FLAGS_RESET 0x8


typedef struct nvtxMemPermissionsAssignRegionDesc_v1
{
    uint32_t flags; /* NVTX_MEM_PERMISSIONS_REGION_FLAGS_* */
    uint32_t regionRefType; /* NVTX_MEM_REGION_REF_TYPE_* */
    nvtxMemRegionRef_t region;

} nvtxMemPermissionsAssignRegionDesc_v1 ;
typedef nvtxMemPermissionsAssignRegionDesc_v1 nvtxMemPermissionsAssignRegionDesc_t;


typedef struct nvtxMemPermissionsAssignBatch_v1
{
    uint16_t extCompatID; /* Set to NVTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t reserved0;

    nvtxMemPermissionsHandle_t permissions;

    size_t regionCount;
    size_t regionElementSize;
    nvtxMemPermissionsAssignRegionDesc_t const* regionElements;

    size_t reserved1;
} nvtxMemPermissionsAssignBatch_v1 ;
typedef nvtxMemPermissionsAssignBatch_v1 nvtxMemPermissionsAssignBatch_t;


 /** \brief Change the permissions of a region of process virtual memory. */
NVTX_DECLSPEC void NVTX_API nvtxMemPermissionsAssign(
    nvtxDomainHandle_t domain,
    nvtxMemPermissionsAssignBatch_t const* desc);


/**
 * \brief Create a permissions object for fine grain thread-local control in
 * multi-threading scenarios
 *
 * Unlike the global permissions object (NVTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE), a new
 * permissions object is empty. There are no regions registered to it, so more memory is accessible
 * if bound(bind) without calls to nvtxMemPermissionsSetAccess* first. The permissions are not
 * active until nvtxMemPermissionsBind. See `nvtxMemPermissionsBind` for more details.
 *
 * Use the flags NVTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_* to control  how the regions in
 * this permission object will interact with global permissions when bound. You may choose to
 * either replace global memory regions setting or overlay on top of them. The most common uses are
 * as follows:
 *     * To limit tools to validate writing exclusively specified in this object but inherit all
 *       global read access regions use `NVTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_WRITE`
 *     * To limit tools to validate both read & write permissions exclusively specified in this
 *        object use NVTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_READ
 *                   & NVTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_WRITE
 *
 * Also see `nvtxMemPermissionsBind` & `nvtxMemPermissionsSetAccess*`.
 */
NVTX_DECLSPEC nvtxMemPermissionsHandle_t NVTX_API nvtxMemPermissionsCreate(
    nvtxDomainHandle_t domain,
    int32_t creationflags); /* NVTX_MEM_PERMISSIONS_CREATE_FLAGS_* */

/**
 * \brief Destroy the permissions object.
 *
 * If bound(bind), destroy will also unbind it.
 */
NVTX_DECLSPEC void NVTX_API nvtxMemPermissionsDestroy(
    nvtxDomainHandle_t domain,
    nvtxMemPermissionsHandle_t permissionsHandle); /* only supported on objects from nvtxMemPermissionsCreate */

/** \brief Reset the permissions object back to its created state. */
NVTX_DECLSPEC void NVTX_API nvtxMemPermissionsReset(
    nvtxDomainHandle_t domain,
    nvtxMemPermissionsHandle_t permissionsHandle);
/* NVTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE and other special handles are supported */


#define NVTX_MEM_PERMISSIONS_BIND_FLAGS_NONE 0x0

 /** \brief Upon binding, with the thread, exclude parent scope write regions instead of overlaying on top of them.
  *
   * EX A developer may chose to first prevent all writes except the ones specified to avoid
  * OOB writes, since there are typically less regions written to than read from.
 **/
#define NVTX_MEM_PERMISSIONS_BIND_FLAGS_STRICT_WRITE 0x2

 /** \brief Upon binding, with the thread, exclude parent scope read regions instead of overlaying on top of them.
  *
  * EX After eliminating any errors when applying strict writes, a developer may then choose to
  * annotate and enforce strict reads behaviors in segments of code.
 **/
#define NVTX_MEM_PERMISSIONS_BIND_FLAGS_STRICT_READ 0x1

 /** \brief Upon binding, with the thread, exclude parent scope atomic RW regions instead of overlaying on top of them.
  *
  * EX After eliminating any errors from read and write, a developer may chose to ensure
  * that atomics are in their own region, removing standard read/write, and replacing with
  * this strict atomic only access.  This way they know that conventional reads or writes
  * will not cause unepected issues.
 **/
#define NVTX_MEM_PERMISSIONS_BIND_FLAGS_STRICT_ATOMIC 0x4


#define NVTX_MEM_PERMISSIONS_BIND_SCOPE_UNKNOWN 0x0

 /** \brief Bind to thread scope.  In this case, tools should validate that local thread's
  * execution is honoring the permissions as well as the state of NVTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE
  * at the time of binding.  If this is not bound then NVTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE should be
  * used to validate the memory.
  *
  * Not all tools will support every scope, such a GPU sanitizer.
 **/
#define NVTX_MEM_PERMISSIONS_BIND_SCOPE_CPU_THREAD 0x1

/**
 * \brief Bind to CUDA stream scope.
 *
 * In this case, work enqueued to a CUDA stream should be validated by the tool,
 * when it executes, that it respect the permission of the permission at the point
 * of binding, as well as the appropriate nvtxMemCudaGetDevicePermissions at the
 * time of binding. If this is not bound then nvtxMemCudaGetDevicePermissions at
 * the time of stream enqueue should be used to validate the memory.
 *
 * This could apply to work done either on the GPU like a kernel launch or to
 * CPU based callbacks like cudaStreamAddCallback if the tools supports it.
 *
 * Binding is applies locally to a CPU thread so that if N CPU threads are enqueing
 * work to the same stream (like the default stream) that there cannot be a race
 * condition between thread binding vs launching their work. IE users should
 * expect the permissions bound in the thread to be honored by the proceeding
 * work (launches, copies, etc) invoked from in the CPU thread until unbound.
 */
#define NVTX_MEM_PERMISSIONS_BIND_SCOPE_CUDA_STREAM 0x2


/**
 * \brief Bind the permissions object into a particular scope on the caller thread
 *
 * Permissions do not take affect until binding. Binding permissions is a thread local
 * activity that overrides global behaviors.  This is to avoid multi-threaded race conditions,
 *
 * The scope dictates what type of processing it applies to, and when in some cases.
 * EX1: NVTX_MEM_PERMISSIONS_BIND_SCOPE_CPU_THREAD applies to CPU code accessing memory while bound.
 * EX2: NVTX_MEM_PERMISSIONS_BIND_SCOPE_CUDA_STREAM applies to CUDA streams, and the permissions
 * must be recorded and applied when the work in the stream dequeues to executes.  In this case
 * it could be GPU or CPU, if the tool support both.
 *
 * Bind can be called again on the same object and thread to take any updates to the
 * specified permission object or the inherited properties.
 *
 * Bind flags support changing how the binding process inherits region access control.
 * In the case of thread scope this is NVTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE and from CUDA_STREAM
 * this is nvtxMemCudaGetDevicePermissions.  Choosing stricter modes allows the user to
 * further reduce the access with less work, since memory by default, behaves as natural
 * until the NVTX annotations instructs a tool to treat it anther way.  See strict flags
 * for more details.
 *
 * Also see nvtxMemPermissionsUnbind
 */
NVTX_DECLSPEC void NVTX_API nvtxMemPermissionsBind(
    nvtxDomainHandle_t domain,
    nvtxMemPermissionsHandle_t permissions, /* special object like NVTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE are not supported */
    uint32_t bindScope, /* NVTX_MEM_PERMISSIONS_BIND_SCOPE_* */
    uint32_t bindFlags); /* NVTX_MEM_PERMISSIONS_BIND_FLAGS_* */

/**
 * \brief Unbind the permissions object bound to the caller thread.
 *
 * Upon unbind, the thread local permissions for a scope are restored to the default
 * behavior defined by the scope.
 */
NVTX_DECLSPEC void NVTX_API nvtxMemPermissionsUnbind(
    nvtxDomainHandle_t domain,
    uint32_t bindScope);

/** @} */ /*END defgroup*/

typedef enum NvtxExtMemCallbackId
{
    /* CBID 0 is invalid */
    NVTX3EXT_CBID_nvtxMemHeapRegister                  = 1,
    NVTX3EXT_CBID_nvtxMemHeapUnregister                = 2,
    NVTX3EXT_CBID_nvtxMemHeapReset                     = 3,
    NVTX3EXT_CBID_nvtxMemRegionsRegister               = 4,
    NVTX3EXT_CBID_nvtxMemRegionsResize                 = 5,
    NVTX3EXT_CBID_nvtxMemRegionsUnregister             = 6,
    NVTX3EXT_CBID_nvtxMemRegionsName                   = 7,
    NVTX3EXT_CBID_nvtxMemPermissionsAssign             = 8,
    NVTX3EXT_CBID_nvtxMemPermissionsCreate             = 9,
    NVTX3EXT_CBID_nvtxMemPermissionsDestroy            = 10,
    NVTX3EXT_CBID_nvtxMemPermissionsReset              = 11,
    NVTX3EXT_CBID_nvtxMemPermissionsBind               = 12,
    NVTX3EXT_CBID_nvtxMemPermissionsUnbind             = 13,

    /* 14-16 in nvtExtImplMemCudaRt1.h */
    NVTX3EXT_CBID_nvtxMemCudaGetProcessWidePermissions = 14,
    NVTX3EXT_CBID_nvtxMemCudaGetDeviceWidePermissions  = 15,
    NVTX3EXT_CBID_nvtxMemCudaSetPeerAccess             = 16,

    NVTX3EXT_CBID_MEM_FN_NUM                           = 17
} NvtxExtMemCallbackId;

#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

/* Extension types are required for the implementation and the NVTX handler. */
#define NVTX_EXT_TYPES_GUARD /* Ensure other headers cannot be included directly */
#include "nvtxDetail/nvtxExtTypes.h"
#undef NVTX_EXT_TYPES_GUARD

#ifndef NVTX_NO_IMPL
/* Ensure other headers cannot be included directly */
#define NVTX_EXT_IMPL_MEM_GUARD
#include "nvtxDetail/nvtxExtImplMem_v1.h"
#undef NVTX_EXT_IMPL_MEM_GUARD
#endif /*NVTX_NO_IMPL*/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NVTOOLSEXTV3_MEM_V1 */
