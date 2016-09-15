program test
use iso_c_binding
use iso_fortran_env
use cudaFor
use ncclFor
implicit none
integer(int32) :: stat, i
real(real32) :: err
integer(int32) :: nEl, nDev, root
type(ncclDataType) :: dataType
type(ncclComm), allocatable :: comm(:)
type(c_ptr), allocatable :: commPtr(:)
integer(int32), allocatable :: devList(:)
type(ncclResult) :: res
integer(int32) :: cudaDev, rank
integer(cuda_stream_kind), allocatable :: stream(:)
real(real32), allocatable :: hostBuff(:, :)
real(real32), allocatable, device :: devBuff(:)
type(c_devptr), allocatable :: devBuffPtr(:)

  nEl = 2621440

!  nDev = 2
!  root = 0
  stat = cudaGetDeviceCount(nDev)
  root = nDev - 1

  dataType = ncclFloat

  allocate(comm(nDev))
  allocate(commPtr(nDev))
  allocate(devList(nDev))

  do i = 1, nDev
    commPtr(i) = c_loc(comm(i))
    devList(i) = i - 1
  end do

  res = ncclCommInitAll(commPtr, nDev, devList)

  do i = 1, nDev
    res = ncclCommCuDevice(commPtr(i), cudaDev)
    res = ncclCommUserRank(commPtr(i), rank)
  end do

  allocate(stream(nDev))

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaStreamCreate(stream(i))
  end do

  allocate(hostBuff(nEl, nDev + 1))

  call random_number(hostBuff(:, 1:nDev))

  hostBuff(:, nDev + 1) = hostBuff(:, root + 1)

  print "(a)", "before broadcast:"
  do i = 1, nDev
    err = maxval(abs(hostBuff(:, i) / hostBuff(:, nDev + 1) - 1.0_real32))
    print "(a, i2.2, a, i2.2, a, e10.4e2)", "maximum error of rank ", i - 1, " vs root (rank ", root,") = ", err
  end do

  allocate(devBuffPtr(nDev))

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    allocate(devBuff(nEl))
    devBuffPtr(i) = c_devloc(devBuff)
    devBuff = hostBuff(:, i)
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    res = ncclBCast(devBuffPtr(i), nEl, dataType, root, commPtr(i), stream(i))
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaStreamSynchronize(stream(i))
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    call c_f_pointer(devBuffPtr(i), devBuff, [nEl])
    hostBuff(:, i) = devBuff
  end do

  print "(a)", ""
  print "(a)", "after broadcast:"
  do i = 1, nDev
    err = maxval(abs(hostBuff(:, i) / hostBuff(:, nDev + 1) - 1.0_real32))
    print "(a, i2.2, a, i2.2, a, e10.4e2)", "maximum error of rank ", i - 1, " vs root (rank ", root,") = ", err
  end do
  print "(a)", ""

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    call c_f_pointer(devBuffPtr(i), devBuff, [nEl])
    deallocate(devBuff)
  end do

  deallocate(devBuffPtr)

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaStreamDestroy(stream(i))
  end do

  deallocate(stream)

  do i = 1, nDev
    call ncclCommDestroy(commPtr(i))
  end do

  deallocate(hostBuff)

  deallocate(devList)
  deallocate(commPtr)
  deallocate(comm)

end program test
