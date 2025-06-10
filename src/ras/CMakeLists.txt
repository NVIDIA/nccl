# Create a static library target named "ncclras"
add_executable(
  ncclras

  # Header files
  ras_internal.h  # Private! (as `libnccl` compiles even without `libncclras`)

  # Source files
  client_support.cc
  client.cc
  collectives.cc
  peers.cc
  ras.cc
  rasnet.cc
)

# Link ncclras privately (important! as it will crash with `ld.lld: error: undefined symbol: cudaDriverGetVersion` if not) to nccl
target_link_libraries(ncclras PRIVATE nccl)
