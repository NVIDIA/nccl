# Create a static library target named "ncclras"
#
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

target_link_libraries(ncclras PUBLIC nccl)