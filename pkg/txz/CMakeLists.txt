# TXZ package building for NCCL

find_program(SED_EXECUTABLE sed)

# Set package directories
set(TXZPREPDIR "${CMAKE_BINARY_DIR}/txz")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/txz")
set(TXZGEN_IN create_txz.sh.in)
set(TXZGEN create_txz.sh)
set(TXZTARGETS ${TXZPREPDIR}/${TXZGEN})

# Get package architecture
set(PKG_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Custom target for txz package preparation
add_custom_target(txz_prep ALL
  # Create txz prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TXZPREPDIR}
  COMMENT "Preparing TXZ package files"
)

# Generate create_txz.sh from template
add_custom_command(
  TARGET txz_prep
  COMMAND ${SED_EXECUTABLE} -e 's/\\$$\{nccl:Major\}/${NCCL_MAJOR}/g'
                            -e 's/\\$$\{nccl:Minor\}/${NCCL_MINOR}/g'
                            -e 's/\\$$\{nccl:Patch\}/${NCCL_PATCH}/g'
                            -e 's/\\$$\{nccl:Suffix\}/${NCCL_SUFFIX}/g'
                            -e 's/\\$$\{cuda:Major\}/${CUDA_MAJOR}/g'
                            -e 's/\\$$\{cuda:Minor\}/${CUDA_MINOR}/g'
                            -e 's/\\$$\{pkg:Revision\}/${PKG_REVISION}/g'
                            -e 's/\\$$\{pkg:Arch\}/${PKG_ARCH}/g'
            ${CMAKE_CURRENT_SOURCE_DIR}/${TXZGEN_IN} > ${TXZPREPDIR}/${TXZGEN}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${TXZGEN_IN}
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
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${TXZPREPDIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning TXZ package build files"
)

# Add txz targets to the main project
add_dependencies(txz_build nccl_static)
