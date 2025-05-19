# Create a static library target named "ncclras"
add_library(ncclras STATIC
  # Header files
  ras_internal.h

  # Source files
  client_support.cc
  client.cc
  collectives.cc
  peers.cc
  ras.cc
  rasnet.cc
)
