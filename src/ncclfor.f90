!Start defines
!End defines

!Start nccl module
module ncclFor
use iso_c_binding
use cudaFor
implicit none
private
public :: ncclResult,                 &
          ncclSuccess,                &
          ncclUnhandledCudaError,     &
          ncclSystemError,            &
          ncclInternalError,          &
          ncclInvalidDevicePointer,   &
          ncclInvalidRank,            &
          ncclUnsupportedDeviceCount, &
          ncclDeviceNotFound,         &
          ncclInvalidDeviceIndex,     &
          ncclLibWrapperNotSet,       &
          ncclCudaMallocFailed,       &
          ncclRankMismatch,           &
          ncclInvalidArgument,        &
          ncclInvalidType,            &
          ncclInvalidOperation,       &
          nccl_NUM_RESULTS
public :: ncclDataType, &
          ncclChar,     &
          ncclInt,      &
#ifdef CUDA_HAS_HALF
          ncclHalf,     &
#endif
          ncclFloat,    &
          ncclDouble,   &
          ncclInt64,    &
          ncclUInt64,   &
          nccl_NUM_TYPES
public :: ncclRedOp, &
          ncclSum,   &
          ncclProd,  &
          ncclMax,   &
          ncclMin,   &
          nccl_NUM_OPS
public :: ncclCommInitAll
public :: ncclCommCuDevice
public :: ncclCommUserRank
public :: ncclCommCount
public :: ncclCommDestroy
public :: ncclReduce
public :: ncclAllReduce
public :: ncclReduceScatter
public :: ncclBCast
public :: ncclAllGather

!Start types

!Start ncclResult
type ncclResult
integer(c_int) :: member
end type ncclResult

type(ncclResult), parameter :: ncclSuccess                = ncclResult( 0), &
                               ncclUnhandledCudaError     = ncclResult( 1), &
                               ncclSystemError            = ncclResult( 2), &
                               ncclInternalError          = ncclResult( 3), &
                               ncclInvalidDevicePointer   = ncclResult( 4), &
                               ncclInvalidRank            = ncclResult( 5), &
                               ncclUnsupportedDeviceCount = ncclResult( 6), &
                               ncclDeviceNotFound         = ncclResult( 7), &
                               ncclInvalidDeviceIndex     = ncclResult( 8), &
                               ncclLibWrapperNotSet       = ncclResult( 9), &
                               ncclCudaMallocFailed       = ncclResult(10), &
                               ncclRankMismatch           = ncclResult(11), &
                               ncclInvalidArgument        = ncclResult(12), &
                               ncclInvalidType            = ncclResult(13), &
                               ncclInvalidOperation       = ncclResult(14), &
                               nccl_NUM_RESULTS           = ncclResult(15)
!End ncclResult

!Start ncclDataType
type ncclDataType
integer(c_int) :: member
end type ncclDataType

type(ncclDataType), parameter :: ncclChar       = ncclDataType(0), &
                                 ncclInt        = ncclDataType(1), &
#ifdef CUDA_HAS_HALF
                                 ncclHalf       = ncclDataType(2), &
#endif
                                 ncclFloat      = ncclDataType(3), &
                                 ncclDouble     = ncclDataType(4), &
                                 ncclInt64      = ncclDataType(5), &
                                 ncclUInt64     = ncclDataType(6), &
                                 nccl_NUM_TYPES = ncclDataType(7)
!End ncclDataType

!Start ncclRedOp
type ncclRedOp
integer(c_int) :: member
end type ncclRedOp

type(ncclRedOp), parameter :: ncclSum      = ncclRedOp(0), &
                              ncclProd     = ncclRedOp(1), &
                              ncclMax      = ncclRedOp(2), &
                              ncclMin      = ncclRedOp(3), &
                              nccl_NUM_OPS = ncclRedOp(4)
!End ncclRedOp

!End types

!Start interfaces
interface

!Start ncclCommInitAll
type(ncclResult) function ncclCommInitAll(comm, ndev, devlist) bind(c, name = 'ncclCommInitAll')
import :: c_ptr, c_int
implicit none
type(c_ptr) :: comm(*)
integer(c_int), value :: ndev
integer(c_int) :: devlist(*)
end function ncclCommInitAll
!End ncclCommInitAll

!Start ncclCommCuDevice
type(ncclResult) function ncclCommCuDevice(comm, devid) bind(c, name = 'ncclCommCuDevice')
import :: c_ptr, c_int
implicit none
type(c_ptr), value :: comm
integer(c_int) :: devid
end function ncclCommCuDevice
!End ncclCommCuDevice

!Start ncclCommUserRank
type(ncclResult) function ncclCommUserRank(comm, rank) bind(c, name = 'ncclCommUserRank')
import :: c_ptr, c_int
implicit none
type(c_ptr), value :: comm
integer(c_int) :: rank
end function ncclCommUserRank
!End ncclCommUserRank

!Start ncclCommCount
type(ncclResult) function ncclCommCount(comm, count) bind(c, name = 'ncclCommCount')
import :: c_ptr, c_int
implicit none
type(c_ptr), value :: comm
integer(c_int) :: count
end function ncclCommCount
!End ncclCommCount

!Start ncclCommDestroy
subroutine ncclCommDestroy(comm) bind(c, name = 'ncclCommDestroy')
import :: c_ptr
implicit none
type(c_ptr), value :: comm
end subroutine ncclCommDestroy
!End ncclCommDestroy

!Start ncclReduce
type(ncclResult) function ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream) bind(c, name = 'ncclReduce')
import :: c_devptr, c_int, c_ptr
import :: ncclDataType, ncclRedOp
import :: cuda_stream_kind
implicit none
type(c_devptr), value :: sendbuff
type(c_devptr), value :: recvbuff
integer(c_int), value :: count
type(ncclDataType), value :: datatype
type(ncclRedOp), value :: op
integer(c_int), value :: root
type(c_ptr), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclReduce
!End ncclReduce

!Start ncclAllReduce
type(ncclResult) function ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream) bind(c, name = 'ncclAllReduce')
import :: c_devptr, c_int, c_ptr
import :: ncclDataType, ncclRedOp
import :: cuda_stream_kind
implicit none
type(c_devptr), value :: sendbuff
type(c_devptr), value :: recvbuff
integer(c_int), value :: count
type(ncclDataType), value :: datatype
type(ncclRedOp), value :: op
type(c_ptr), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclAllReduce
!End ncclAllReduce

!Start ncclReduceScatter
type(ncclResult) function ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream) bind(c, name = 'ncclReduceScatter')
import :: c_devptr, c_int, c_ptr
import :: ncclDataType, ncclRedOp
import :: cuda_stream_kind
implicit none
type(c_devptr), value :: sendbuff
type(c_devptr), value :: recvbuff
integer(c_int), value :: recvcount
type(ncclDataType), value :: datatype
type(ncclRedOp), value :: op
type(c_ptr), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclReduceScatter
!End ncclReduceScatter

!Start ncclBCast
type(ncclResult) function ncclBCast(buff, count, datatype, root, comm, stream) bind(c, name = 'ncclBcast')
import :: c_devptr, c_int, c_ptr
import :: ncclDataType
import :: cuda_stream_kind
implicit none
type(c_devptr), value :: buff
integer(c_int), value :: count
type(ncclDataType), value :: datatype
integer(c_int), value :: root
type(c_ptr), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclBCast
!End ncclBCast

!Start ncclAllGather
type(ncclResult) function ncclAllGather(sendbuff, count, datatype, recvbuff, comm, stream) bind(c, name = 'ncclAllGather')
import :: c_devptr, c_int, c_ptr
import :: ncclDataType
import :: cuda_stream_kind
implicit none
type(c_devptr), value :: sendbuff
integer(c_int), value :: count
type(ncclDataType), value :: datatype
type(c_devptr), value :: recvbuff
type(c_ptr), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclAllGather
!End ncclAllGather

end interface
!End interfaces

end module ncclFor
!End nccl module
