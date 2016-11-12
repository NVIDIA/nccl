#ifndef _CUDA

!Start cudaFor module
module cudaFor
use iso_c_binding
implicit none
private
public :: c_devptr
public :: cudaMemCpyKind,           &
          cudaMemCpyHostToHost,     &
          cudaMemCpyHostToDevice,   &
          cudaMemCpyDeviceToHost,   &
          cudaMemCpyDeviceToDevice, &
          cudaMemCpyDefault
public :: cuda_stream_kind
public :: cudaGetDeviceCount
public :: cudaSetDevice
public :: cudaMAlloc
public :: cudaMemCpy
public :: cudaFree
public :: cudaStreamCreate
public :: cudaStreamSynchronize
public :: cudaStreamDestroy

!Start types

!Start c_devptr
type, bind(c) :: c_devptr
type(c_ptr) :: member
end type c_devptr
!End c_devptr

!Start cudaMemCpyKind
type, bind(c) :: cudaMemCpyKind
integer(c_int) :: member
end type cudaMemCpyKind

type(cudaMemCpyKind), parameter :: cudaMemCpyHostToHost     = cudaMemCpyKind(0), &
                                   cudaMemCpyHostToDevice   = cudaMemCpyKind(1), &
                                   cudaMemCpyDeviceToHost   = cudaMemCpyKind(2), &
                                   cudaMemCpyDeviceToDevice = cudaMemCpyKind(3), &
                                   cudaMemCpyDefault        = cudaMemCpyKind(4)
!End cudaMemCpyKind

!Start cuda_stream_kind
integer(c_intptr_t), parameter :: cuda_stream_kind = c_intptr_t
!End cuda_stream_kind

!End types

!Start interfaces

!Start cudaGetDeviceCount
interface cudaGetDeviceCount
integer(c_int) function cudaGetDeviceCount(count) bind(c, name = "cudaGetDeviceCount")
import :: c_int
implicit none
integer(c_int) :: count
end function cudaGetDeviceCount
end interface cudaGetDeviceCount
!End cudaGetDeviceCount

!Start cudaSetDevice
interface cudaSetDevice
integer(c_int) function cudaSetDevice(device) bind(c, name = "cudaSetDevice")
import :: c_int
implicit none
integer(c_int), value :: device
end function cudaSetDevice
end interface cudaSetDevice
!End cudaSetDevice

!Start cudaMAlloc
interface cudaMAlloc
integer(c_int) function cudaMAlloc(devPtr, size) bind(c, name = "cudaMalloc")
import :: c_int, c_size_t
import :: c_devptr
implicit none
type(c_devptr) :: devPtr
integer(c_size_t), value :: size
end function cudaMAlloc
end interface cudaMAlloc
!End cudaMAlloc

!Start cudaMemCpy
interface cudaMemCpy

!Start cudaMemCpyH2D
integer(c_int) function cudaMemCpyH2D(dst, src, count, kind) bind(c, name = "cudaMemcpy")
import :: c_ptr, c_int, c_size_t
import :: c_devptr, cudaMemCpyKind
implicit none
type(c_devptr), value :: dst
type(c_ptr), value :: src
integer(c_size_t), value :: count
type(cudaMemCpyKind), value :: kind
end function cudaMemCpyH2D
!End cudaMemCpyH2D

!Start cudaMemCpyD2H
integer(c_int) function cudaMemCpyD2H(dst, src, count, kind) bind(c, name = "cudaMemcpy")
import :: c_ptr, c_int, c_size_t
import :: c_devptr, cudaMemCpyKind
implicit none
type(c_ptr), value :: dst
type(c_devptr), value :: src
integer(c_size_t), value :: count
type(cudaMemCpyKind), value :: kind
end function cudaMemCpyD2H
!End cudaMemCpyD2H

end interface cudaMemCpy
!End cudaMemCpy

!Start cudaFree
interface cudaFree
integer(c_int) function cudaFree(devPtr) bind(c, name = "cudaFree")
import :: c_int
import :: c_devptr
implicit none
type(c_devptr), value :: devPtr
end function cudaFree
end interface cudaFree
!End cudaFree

!Start cudaStreamCreate
interface cudaStreamCreate
integer(c_int) function cudaStreamCreate(pStream) bind(c, name = "cudaStreamCreate")
import :: c_int
import :: cuda_stream_kind
implicit none
integer(cuda_stream_kind) :: pStream
end function cudaStreamCreate
end interface cudaStreamCreate
!End cudaStreamCreate

!Start cudaStreamSynchronize
interface cudaStreamSynchronize
integer(c_int) function cudaStreamSynchronize(stream) bind(c, name = "cudaStreamSynchronize")
import :: c_int
import :: cuda_stream_kind
implicit none
integer(cuda_stream_kind), value :: stream
end function cudaStreamSynchronize
end interface cudaStreamSynchronize
!End cudaStreamSynchronize

!Start cudaStreamDestroy
interface cudaStreamDestroy
integer(c_int) function cudaStreamDestroy(stream) bind(c, name = "cudaStreamDestroy")
import :: c_int
import :: cuda_stream_kind
implicit none
integer(cuda_stream_kind), value :: stream
end function cudaStreamDestroy
end interface cudaStreamDestroy
!End cudaStreamDestroy

!End interfaces

end module cudaFor
!End cudaFor module

#endif
