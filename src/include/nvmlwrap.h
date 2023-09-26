/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NVMLWRAP_H_
#define NCCL_NVMLWRAP_H_

#include "nccl.h"

//#define NCCL_NVML_DIRECT 1
#ifndef NCCL_NVML_DIRECT
#define NCCL_NVML_DIRECT 0
#endif

#if NCCL_NVML_DIRECT
#include "nvml.h"
#else
// Dynamically handle dependencies on NVML

/* Extracted from nvml.h */
typedef struct nvmlDevice_st* nvmlDevice_t;
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE   16

typedef enum nvmlEnableState_enum
{
    NVML_FEATURE_DISABLED    = 0,     //!< Feature disabled
    NVML_FEATURE_ENABLED     = 1      //!< Feature enabled
} nvmlEnableState_t;

typedef enum nvmlNvLinkCapability_enum
{
    NVML_NVLINK_CAP_P2P_SUPPORTED = 0,     // P2P over NVLink is supported
    NVML_NVLINK_CAP_SYSMEM_ACCESS = 1,     // Access to system memory is supported
    NVML_NVLINK_CAP_P2P_ATOMICS   = 2,     // P2P atomics are supported
    NVML_NVLINK_CAP_SYSMEM_ATOMICS= 3,     // System memory atomics are supported
    NVML_NVLINK_CAP_SLI_BRIDGE    = 4,     // SLI is supported over this link
    NVML_NVLINK_CAP_VALID         = 5,     // Link is supported on this device
    // should be last
    NVML_NVLINK_CAP_COUNT
} nvmlNvLinkCapability_t;

typedef enum nvmlReturn_enum
{
    NVML_SUCCESS = 0,                   //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,       //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,    //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,       //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,       //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5, //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,           //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,   //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,  //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,   //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,            //!< User provided timeout passed
    NVML_ERROR_IRQ_ISSUE = 11,          //!< NVIDIA Kernel detected an interrupt issue with a GPU
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,  //!< NVML Shared Library couldn't be found or loaded
    NVML_ERROR_FUNCTION_NOT_FOUND = 13, //!< Local version of NVML doesn't implement this function
    NVML_ERROR_CORRUPTED_INFOROM = 14,  //!< infoROM is corrupted
    NVML_ERROR_GPU_IS_LOST = 15,        //!< The GPU has fallen off the bus or has otherwise become inaccessible
    NVML_ERROR_RESET_REQUIRED = 16,     //!< The GPU requires a reset before it can be used again
    NVML_ERROR_OPERATING_SYSTEM = 17,   //!< The GPU control device has been blocked by the operating system/cgroups
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    NVML_ERROR_IN_USE = 19,             //!< An operation cannot be performed because the GPU is currently in use
    NVML_ERROR_UNKNOWN = 999            //!< An internal driver error occurred
} nvmlReturn_t;

typedef struct nvmlPciInfo_st
{
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id

    // Added in NVML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID

    // NVIDIA reserved for internal use only
    unsigned int reserved0;
    unsigned int reserved1;
    unsigned int reserved2;
    unsigned int reserved3;
} nvmlPciInfo_t;

/* P2P Capability Index Status*/
typedef enum nvmlGpuP2PStatus_enum
{
    NVML_P2P_STATUS_OK     = 0,
    NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED,
    NVML_P2P_STATUS_GPU_NOT_SUPPORTED,
    NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED,
    NVML_P2P_STATUS_DISABLED_BY_REGKEY,
    NVML_P2P_STATUS_NOT_SUPPORTED,
    NVML_P2P_STATUS_UNKNOWN
} nvmlGpuP2PStatus_t;

/* P2P Capability Index*/
typedef enum nvmlGpuP2PCapsIndex_enum
{
    NVML_P2P_CAPS_INDEX_READ = 0,
    NVML_P2P_CAPS_INDEX_WRITE,
    NVML_P2P_CAPS_INDEX_NVLINK,
    NVML_P2P_CAPS_INDEX_ATOMICS,
    NVML_P2P_CAPS_INDEX_PROP,
    NVML_P2P_CAPS_INDEX_UNKNOWN
} nvmlGpuP2PCapsIndex_t;

/**
 * Represents the type for sample value returned
 */
typedef enum nvmlValueType_enum
{
    NVML_VALUE_TYPE_DOUBLE = 0,
    NVML_VALUE_TYPE_UNSIGNED_INT = 1,
    NVML_VALUE_TYPE_UNSIGNED_LONG = 2,
    NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3,
    NVML_VALUE_TYPE_SIGNED_LONG_LONG = 4,

    // Keep this last
    NVML_VALUE_TYPE_COUNT
}nvmlValueType_t;


/**
 * Union to represent different types of Value
 */
typedef union nvmlValue_st
{
    double dVal;                    //!< If the value is double
    unsigned int uiVal;             //!< If the value is unsigned int
    unsigned long ulVal;            //!< If the value is unsigned long
    unsigned long long ullVal;      //!< If the value is unsigned long long
    signed long long sllVal;        //!< If the value is signed long long
}nvmlValue_t;

/**
 * Field Identifiers.
 *
 * All Identifiers pertain to a device. Each ID is only used once and is guaranteed never to change.
 */

/* NVLink Speed */
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON 90  //!< Common NVLink Speed in MBps for active links
#define NVML_FI_DEV_NVLINK_LINK_COUNT        91  //!< Number of NVLinks present on the device

/**
 * Remote device NVLink ID
 *
 * Link ID needs to be specified in the scopeId field in nvmlFieldValue_t.
 */
#define NVML_FI_DEV_NVLINK_REMOTE_NVLINK_ID     146 //!< Remote device NVLink ID

/**
 * NVSwitch: connected NVLink count
 */
#define NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT   147  //!< Number of NVLinks connected to NVSwitch

#define NVML_FI_DEV_NVLINK_GET_SPEED                  164
#define NVML_FI_DEV_NVLINK_GET_STATE                  165
#define NVML_FI_DEV_NVLINK_GET_VERSION                166

#define NVML_FI_DEV_C2C_LINK_COUNT                    170 //!< Number of C2C Links present on the device
#define NVML_FI_DEV_C2C_LINK_GET_STATUS               171 //!< C2C Link Status 0=INACTIVE 1=ACTIVE
#define NVML_FI_DEV_C2C_LINK_GET_MAX_BW               172 //!< C2C Link Speed in MBps for active links

#define NVML_FI_MAX 173 //!< One greater than the largest field ID defined above

/**
 * Information for a Field Value Sample
 */
typedef struct nvmlFieldValue_st
{
    unsigned int fieldId;       //!< ID of the NVML field to retrieve. This must be set before any call that uses this struct. See the constants starting with NVML_FI_ above.
    unsigned int scopeId;       //!< Scope ID can represent data used by NVML depending on fieldId's context. For example, for NVLink throughput counter data, scopeId can represent linkId.
    long long timestamp;        //!< CPU Timestamp of this value in microseconds since 1970
    long long latencyUsec;      //!< How long this field value took to update (in usec) within NVML. This may be averaged across several fields that are serviced by the same driver call.
    nvmlValueType_t valueType;  //!< Type of the value stored in value
    nvmlReturn_t nvmlReturn;    //!< Return code for retrieving this value. This must be checked before looking at value, as value is undefined if nvmlReturn != NVML_SUCCESS
    nvmlValue_t value;          //!< Value for this field. This is only valid if nvmlReturn == NVML_SUCCESS
} nvmlFieldValue_t;

/* End of nvml.h */
#endif // NCCL_NVML_DIRECT

constexpr int ncclNvmlMaxDevices = 32;
struct ncclNvmlDeviceInfo {
  nvmlDevice_t handle;
  int computeCapabilityMajor, computeCapabilityMinor;
};
struct ncclNvmlDevicePairInfo {
  nvmlGpuP2PStatus_t p2pStatusRead, p2pStatusWrite;
};
extern int ncclNvmlDeviceCount;
extern ncclNvmlDeviceInfo ncclNvmlDevices[ncclNvmlMaxDevices];
extern ncclNvmlDevicePairInfo ncclNvmlDevicePairs[ncclNvmlMaxDevices][ncclNvmlMaxDevices];

// All ncclNvmlFoo() functions call ncclNvmlEnsureInitialized() implicitly.
// Outsiders need only call it if they want to inspect the ncclNvml global
// tables above.
ncclResult_t ncclNvmlEnsureInitialized();

ncclResult_t ncclNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device);
ncclResult_t ncclNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index);
ncclResult_t ncclNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
ncclResult_t ncclNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
ncclResult_t ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
ncclResult_t ncclNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult);
ncclResult_t ncclNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor);
ncclResult_t ncclNvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus);
ncclResult_t ncclNvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values);

#endif // End include guard
