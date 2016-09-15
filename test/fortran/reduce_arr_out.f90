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
type(ncclRedOp) :: redOp
type(ncclComm), allocatable :: comm(:)
type(c_ptr), allocatable :: commPtr(:)
integer(int32), allocatable :: devList(:)
type(ncclResult) :: res
integer(int32) :: cudaDev, rank
integer(cuda_stream_kind), allocatable :: stream(:)
real(real32), allocatable :: hostBuff(:, :)
real(real32), allocatable, device :: sendBuff(:)
type(c_devptr), allocatable :: sendBuffPtr(:)
real(real32), allocatable, device :: recvBuff(:)
type(c_devptr), allocatable :: recvBuffPtr(:)

  nEl = 2621440

!  nDev = 2
!  root = 0
  stat = cudaGetDeviceCount(nDev)
  root = nDev - 1

  dataType = ncclFloat
  redOp = ncclProd

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

  allocate(hostBuff(nEl, nDev + 2))

  call random_number(hostBuff(:, 1:nDev + 1))

  hostBuff(:, nDev + 2) = hostBuff(:, 1)
  do i = 2, nDev
    hostBuff(:, nDev + 2) = hostBuff(:, nDev + 2) * hostBuff(:, i)
  end do

  print "(a)", "before reduce:"
  err = maxval(abs(hostBuff(:, nDev + 1) / hostBuff(:, nDev + 2) - 1.0_real32))
  print "(a, i2.2, a, e10.4e2)", "maximum error in recvbuff from root (rank ", root,") = ", err

  allocate(sendBuffPtr(nDev))

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    allocate(sendBuff(nEl))
    sendBuffPtr(i) = c_devloc(sendBuff)
    sendBuff = hostBuff(:, i)
  end do

  allocate(recvBuffPtr(nDev))

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    allocate(recvBuff(nEl))
    recvBuffPtr(i) = c_devloc(recvBuff)
    recvBuff = hostBuff(:, i)
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    res = ncclReduce(sendBuffPtr(i), recvBuffPtr(i), nEl, dataType, redOp, root, commPtr(i), stream(i))
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaStreamSynchronize(stream(i))
  end do

  stat = cudaSetDevice(devList(root + 1))
  call c_f_pointer(recvBuffPtr(root + 1), recvBuff, [nEl])
  hostBuff(:, nDev + 1) = recvBuff

  print "(a)", ""
  print "(a)", "after reduce:"
  err = maxval(abs(hostBuff(:, nDev + 1) / hostBuff(:, nDev + 2) - 1.0_real32))
  print "(a, i2.2, a, e10.4e2)", "maximum error in recvbuff from root (rank ", root,") = ", err

  print "(a)", ""
  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    call c_f_pointer(sendBuffPtr(i), sendBuff, [nEl])
    hostBuff(:, nDev + 1) = sendBuff
    err = maxval(abs(hostBuff(:, nDev + 1) / hostBuff(:, i) - 1.0_real32))
    print "(a, i2.2, a, e10.4e2)", "maximum error in sendbuff of rank ", i - 1," = ", err
  end do

  print "(a)", ""
  do i = 1, nDev
    if (i - 1 /= root) then
      stat = cudaSetDevice(devList(i))
      call c_f_pointer(recvBuffPtr(i), recvBuff, [nEl])
      hostBuff(:, nDev + 1) = recvBuff
      err = maxval(abs(hostBuff(:, nDev + 1) / hostBuff(:, i) - 1.0_real32))
      print "(a, i2.2, a, e10.4e2)", "maximum error in recvbuff of rank ", i - 1," = ", err
    end if
  end do
  print "(a)", ""

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    call c_f_pointer(recvBuffPtr(i), recvBuff, [nEl])
    deallocate(recvBuff)
  end do

  deallocate(recvBuffPtr)

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    call c_f_pointer(sendBuffPtr(i), sendBuff, [nEl])
    deallocate(sendBuff)
  end do

  deallocate(sendBuffPtr)

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
