# Create an executable target named "ncclras"
add_executable(
  ncclras

  # @HELP: probably we should refactor that into `target_include_directories()`
  # Header files
  ras_internal.h  # Private! (as `libnccl` compiles even without `libncclras`)

  # Source files
  client.cc

  # NOTE: Below is only used with `libnccl` and are not needed for `ncclras` executable!
  # client_support.cc
  # collectives.cc
  # peers.cc
  # ras.cc
  # rasnet.cc
)

# Link ncclras privately (important! as it will crash with `ld.lld: error: undefined symbol: cudaDriverGetVersion` if not) to nccl
target_link_libraries(ncclras PRIVATE nccl)
# @HELP: check if we need to explicitely link against CUDA libraries or it will transtively propagate from `nccl`
