# Create an executable target named "ncclras"
add_executable(
  ncclras

  # Source files
  client.cc

  # NOTE: Below is only used with `libnccl` and are not needed for `ncclras` executable!
  # client_support.cc
  # collectives.cc
  # peers.cc
  # ras.cc
  # rasnet.cc
)


# Header files - refactored to use target_include_directories()
# ras_internal.h is Private! (as `libnccl` compiles even without `libncclras`)
target_include_directories(ncclras PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(ncclras PRIVATE nccl CUDA::cudart)
# @HELP: check if we need to explicitely link against CUDA libraries or it will transtively propagate from `nccl`
# BEN NOTE: believe we do. it errored with `(cudaMalloc, cudaGetErrorString, cudaGetDevice` blah blah not found