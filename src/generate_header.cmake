
# CMake script to generate nccl.h from nccl.h.in
# This replaces sed usage for Windows compatibility

file(READ ${INPUT_FILE} HEADER_CONTENT)

# Replace placeholders with actual values
string(REPLACE "\${nccl:Major}" "${NCCL_MAJOR}" HEADER_CONTENT "${HEADER_CONTENT}")
string(REPLACE "\${nccl:Minor}" "${NCCL_MINOR}" HEADER_CONTENT "${HEADER_CONTENT}")
string(REPLACE "\${nccl:Patch}" "${NCCL_PATCH}" HEADER_CONTENT "${HEADER_CONTENT}")
string(REPLACE "\${nccl:Suffix}" "${NCCL_SUFFIX}" HEADER_CONTENT "${HEADER_CONTENT}")
string(REPLACE "\${nccl:Version}" "${NCCL_VERSION_CODE}" HEADER_CONTENT "${HEADER_CONTENT}")

file(WRITE ${OUTPUT_FILE} "${HEADER_CONTENT}")

