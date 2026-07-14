# Source TXZ package building for NCCL

# Set package directories
set(TXZPREPDIR "${CMAKE_BINARY_DIR}/srctxz")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/srctxz")
set(TXZGEN_IN create_srctxz.sh.in)
set(TXZGEN create_srctxz.sh)

# Set package revision
set(PKG_REVISION 3)

# Custom target for srctxz package preparation
add_custom_target(srctxz_prep ALL
  # Create srctxz prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TXZPREPDIR}
  COMMENT "Preparing Source TXZ package files"
)

# Generate create_srctxz.sh from its template.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/${TXZGEN_IN}
  ${TXZPREPDIR}/${TXZGEN}
  @ONLY
)

# Build the package
add_custom_target(srctxz_build
  COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/src ${CMAKE_COMMAND} --build . --target clean
  COMMAND cd ${CMAKE_BINARY_DIR} && SRCTXZ_APITESTS=${SRCTXZ_APITESTS} bash srctxz/create_srctxz.sh
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PKGDIR}
  COMMAND mv ${CMAKE_SOURCE_DIR}/../nccl-src*.txz ${PKGDIR}
  WORKING_DIRECTORY ${TXZPREPDIR}
  DEPENDS srctxz_prep
  COMMENT "Building Source TXZ package with create_srctxz.sh"
)

# Custom target for cleaning srctxz package
add_custom_target(srctxz_clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Source TXZ package build files"
)
