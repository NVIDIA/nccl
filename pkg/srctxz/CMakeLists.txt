# Source TXZ package building for NCCL

find_program(SED_EXECUTABLE sed)

# Set package directories
set(TXZPREPDIR "${CMAKE_BINARY_DIR}/srctxz")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/srctxz")
set(TXZGEN_IN create_srctxz.sh.in)
set(TXZGEN create_srctxz.sh)
set(TXZTARGETS ${TXZPREPDIR}/${TXZGEN})

# Set package revision
set(PKG_REVISION 3)
set(PKG_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Custom target for srctxz package preparation
add_custom_target(srctxz_prep ALL
  # Create srctxz prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TXZPREPDIR}
  COMMENT "Preparing Source TXZ package files"
)

# Generate create_srctxz.sh from template
add_custom_command(
  TARGET srctxz_prep
  COMMAND ${SED_EXECUTABLE} -e 's/\\$$\{nccl:Major\}/${NCCL_MAJOR}/g'
                            -e 's/\\$$\{nccl:Minor\}/${NCCL_MINOR}/g'
                            -e 's/\\$$\{nccl:Patch\}/${NCCL_PATCH}/g'
                            -e 's/\\$$\{nccl:Suffix\}/${NCCL_SUFFIX}/g'
                            -e 's/\\$$\{pkg:Revision\}/${PKG_REVISION}/g'
            ${CMAKE_CURRENT_SOURCE_DIR}/${TXZGEN_IN} > ${TXZPREPDIR}/${TXZGEN}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${TXZGEN_IN}
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
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${TXZPREPDIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Source TXZ package build files"
)
