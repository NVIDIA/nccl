.. _api_param:

******************
NCCL Parameter API
******************

The following functions and types provide access to NCCL runtime parameters.
Parameters are identified by string keys (often defined as environment variables,
in a config file, or through the EnvPlugin) and can be read as typed values or
as strings. Two access styles are supported:

- **Handle-based**: look up a parameter by key once and bind it to a :c:type:`ncclParamHandle_t`
  handle with :c:func:`ncclParamBind`, then read typed values through the returned
  handle.
- **Key-based**: look up a parameter by key without a persistent
  handle and read values as strings. These APIs will look up the key for every access.

Types
-----

.. c:type:: ncclParamHandle_t

  Opaque handle that represents a binding between a caller and a 
  parameter. It is created by :c:func:`ncclParamBind`. The handle remains valid for
  duration of the NCCL module's lifetime; callers must not free it or expect it to be
  accessible during static destruction or process tear down.

Handle-Based API
----------------

ncclParamBind
^^^^^^^^^^^^^

.. c:function:: ncclResult_t ncclParamBind(ncclParamHandle_t* out, const char* key)

  Look up the parameter identified by *key* and store a handle to it in
  *\*out*. The returned handle is owned by the parameter system and must not
  be freed by the caller. If *key* does not match any registered parameter,
  the function returns an ``ncclInvalidArgument`` error and *\*out* is left
  unmodified.

  *key* is the canonical name of the parameter (typically the corresponding
  environment variable name).

  Example::

    ncclResult_t ret = ncclSuccess;
    ncclParamHandle_t myHandle = NULL;
    ret = ncclParamBind(&myHandle, "NCCL_IB_ROCE_VERSION_NUM");
    if (ret == ncclSuccess) {
      int32_t roceVerValue = 0;
      ret = ncclParamGetI32(myHandle, &roceVerValue);
      if (ret == ncclSuccess) {
        /* roceVerValue is set to the current value of NCCL_IB_ROCE_VERSION_NUM */
      }
    }

Accessor Family of Functions - ncclResult_t ncclParamGet*(ncclParamHandle_t h, T* out)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:function:: ncclResult_t ncclParamGetI8(ncclParamHandle_t h, int8_t* out)
.. c:function:: ncclResult_t ncclParamGetI16(ncclParamHandle_t h, int16_t* out)
.. c:function:: ncclResult_t ncclParamGetI32(ncclParamHandle_t h, int32_t* out)
.. c:function:: ncclResult_t ncclParamGetI64(ncclParamHandle_t h, int64_t* out)
.. c:function:: ncclResult_t ncclParamGetU8(ncclParamHandle_t h, uint8_t* out)
.. c:function:: ncclResult_t ncclParamGetU16(ncclParamHandle_t h, uint16_t* out)
.. c:function:: ncclResult_t ncclParamGetU32(ncclParamHandle_t h, uint32_t* out)
.. c:function:: ncclResult_t ncclParamGetU64(ncclParamHandle_t h, uint64_t* out)

  Read the value of the parameter bound to *h* as the type T and store it in *\*out*.
  Returns ``ncclInvalidArgument`` if the parameter value cannot be converted to the
  requested type. These functions will also return ``ncclInvalidArgument`` if *h*
  or *out* is ``NULL``.

  This function family comes in all combinations of signed (I) and unsigned (U)
  for 8-, 16-, 32-, and 64-bit integers.

ncclParamGetStr
^^^^^^^^^^^^^^^

.. c:function:: ncclResult_t ncclParamGetStr(ncclParamHandle_t h, const char** out)

  Return a C-style string representation of the parameter bound to *h*. The pointer
  written to *\*out* is owned by the parameter system and is valid until the
  next call to :c:func:`ncclParamGetStr` on the same thread. Callers that need
  to retain the value across calls must copy it.

ncclParamGet
^^^^^^^^^^^^

.. c:function:: ncclResult_t ncclParamGet(ncclParamHandle_t h, void* out, int maxLen, int* len)

  Copy the raw binary representation of the parameter bound to *h* into the
  caller-allocated buffer *out*. *maxLen* is the size of *out* in bytes.
  The number of bytes actually written is returned in *\*len*. If the
  parameter's native representation is larger than *maxLen*, the function
  returns ``ncclInvalidArgument`` and *\*len* is set to the required size.

Key-Based API
----------------

ncclParamGetParameter
^^^^^^^^^^^^^^^^^^^^^

.. c:function:: ncclResult_t ncclParamGetParameter(const char* key, const char** value, int* valueLen)

  Look up the parameter identified by *key* and return its value as a
  null-terminated string. The pointer written to *\*value* is owned by the
  parameter system and is valid until the next call to
  :c:func:`ncclParamGetParameter` on the same thread. *valueLen* receives the
  length of the string in bytes, excluding the null terminator. Returns
  ``ncclInvalidArgument`` if *key* is not found.

  Example::

    ncclResult_t ret = ncclSuccess;
    const char* valueStr = NULL;
    int len = 0;
    ret = ncclParamGetParameter("NCCL_IB_ROCE_VERSION_NUM", &valueStr, &len);
    if (ret == ncclSuccess) {
      printf("NCCL_IB_ROCE_VERSION_NUM=%s\n", valueStr);
    }

ncclParamGetAllParameterKeys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:function:: ncclResult_t ncclParamGetAllParameterKeys(const char*** table, int* tableLen)

  Return a pointer to the internal table of all registered parameter keys.
  *\*table* is set to an array of ``const char*`` pointers, one per
  registered parameter, and *\*tableLen* is set to the number of entries.
  The table and the strings it contains are owned by the parameter system.
  Both are valid until the next call to
  :c:func:`ncclParamGetAllParameterKeys` on the same thread. The keys in the returned
  table are not sorted.

  This function returns parameters published in :doc:`../env` by default. Setting
  ``NCCL_PARAM_DUMP_ALL`` to ``true`` forces this function to return all parameters.

  Example::

    ncclResult_t ret = ncclSuccess;
    const char** keyTable = NULL;
    int tableLen = 0;

    ret = ncclParamGetAllParameterKeys(&keyTable, &tableLen);
    if (ret == ncclSuccess) {
      for (int i = 0; i < tableLen; ++i) {
        printf("Key: %s\n", keyTable[i]); /* prints all published parameter keys */
      }
    }

    putenv("NCCL_PARAM_DUMP_ALL=true");
    ret = ncclParamGetAllParameterKeys(&keyTable, &tableLen);
    if (ret == ncclSuccess) {
      for (int i = 0; i < tableLen; ++i) {
        printf("Key: %s\n", keyTable[i]); /* prints all parameter keys */
      }
    }

ncclParamDumpAll
^^^^^^^^^^^^^^^^

.. c:function:: void ncclParamDumpAll(void)

  Write the key and current value of every registered parameter to the NCCL
  log output. Useful for diagnostics and configuration auditing. The information
  of parameters in the output are not sorted.

  This function returns parameters published in :doc:`../env` by default. Setting
  ``NCCL_PARAM_DUMP_ALL`` to ``true`` forces this function to return all parameters.

  Example of output with ``NCCL_PARAM_DUMP_ALL`` set to ``true``::

    NCCL_NO_CACHE (const char *) [Private, Cached] Comma-separated list of param keys to disable caching (or ALL)
        Current value=<unset> set_by=Default default=<unset>
        Accepted value: String

    NCCL_PARAM_DUMP_ALL (bool) [Private] Print all parameters including private ones
        Current value=TRUE set_by=EnvPlugin default=FALSE
        Accepted value: Boolean: 1/T/TRUE or 0/F/FALSE
