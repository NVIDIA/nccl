# TXZ package building for NCCL

# Set package directories
set(TXZPREPDIR "${CMAKE_BINARY_DIR}/txz")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/txz")
set(TXZGEN_IN create_txz.sh.in)
set(TXZGEN create_txz.sh)

# Get package architecture
set(PKG_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Custom target for txz package preparation
add_custom_target(txz_prep ALL
  # Create txz prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TXZPREPDIR}
  COMMENT "Preparing TXZ package files"
)

# Generate create_txz.sh from its template.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/${TXZGEN_IN}
  ${TXZPREPDIR}/${TXZGEN}
  @ONLY
)

# Build the package
add_custom_target(txz_build
  COMMAND cd ${CMAKE_BINARY_DIR} && bash txz/create_txz.sh
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PKGDIR}
  COMMAND mv ${CMAKE_SOURCE_DIR}/nccl*.txz ${PKGDIR}
  WORKING_DIRECTORY ${TXZPREPDIR}
  DEPENDS txz_prep nccl_static
  COMMENT "Building TXZ package with create_txz.sh"
)

# Custom target for cleaning txz package
add_custom_target(txz_clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning TXZ package build files"
)

# Add txz targets to the main project
add_dependencies(txz_build nccl_static)
