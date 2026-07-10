**********************************************
User Defined Reduction Operators
**********************************************

The following functions are public APIs exposed by NCCL to create and destroy
custom reduction operators for use in reduction collectives.

ncclRedOpCreatePreMulSum
------------------------

.. c:function:: ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm)

Creates a new reduction operator which pre-multiplies input values by a given
scalar locally before reducing them with peer values via summation. Both the
input values and the scalar are of type *datatype*. For use
only with collectives launched against *comm* and *datatype*. The
*residence* argument indicates whether the memory pointed to by *scalar* should be
dereferenced immediately by the host before this function returns
(ncclScalarHostImmediate), or by the device during execution of the reduction
collective (ncclScalarDevice). Upon return, the newly created operator's handle
is stored in *op*.

ncclRedOpDestroy
----------------

.. c:function:: ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm)

Destroys the reduction operator *op*. The operator must have been created by
ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
destroyed as soon as the last NCCL function which is given that operator returns.
