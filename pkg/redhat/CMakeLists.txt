# Redhat package building for NCCL

find_program(RPMBUILD_EXECUTABLE rpmbuild)
find_program(SED_EXECUTABLE sed)

# Set package directories
set(RPMPREPDIR "${CMAKE_BINARY_DIR}/redhat")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/rpm")
set(RPMGEN_IN nccl.spec.in)
set(RPMGEN nccl.spec)
set(RPMFILES ${RPMGEN})
set(RPMTARGETS ${RPMPREPDIR}/${RPMGEN})

# Get current timestamp in RFC 2822 format
execute_process(COMMAND date -R
                OUTPUT_VARIABLE PKG_TIMESTAMP
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get architecture
set(ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
set(PKG_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Get multiarch tuple
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-multiarch
                OUTPUT_VARIABLE PKG_MULTIARCH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)

if(NOT PKG_MULTIARCH)
  # Hardwire the PKG_MULTIARCH directory as the RHEL6 distribution agnostic compiler (gcc 4.8.3) doesn't set it
  set(PKG_MULTIARCH "${ARCH}-linux-gnu")
endif()

# Custom target for redhat package preparation
add_custom_target(redhat_prep ALL
  # Create redhat prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RPMPREPDIR}
  COMMENT "Preparing Redhat package files"
)

# Generate spec file from template
add_custom_command(
  TARGET redhat_prep
  COMMAND ${SED_EXECUTABLE} -e 's/\\$$\{nccl:Major\}/${NCCL_MAJOR}/g'
                            -e 's/\\$$\{nccl:Minor\}/${NCCL_MINOR}/g'
                            -e 's/\\$$\{nccl:Patch\}/${NCCL_PATCH}/g'
                            -e 's/\\$$\{nccl:Suffix\}/${NCCL_SUFFIX}/g'
                            -e 's/\\$$\{cuda:Major\}/${CUDA_MAJOR}/g'
                            -e 's/\\$$\{cuda:Minor\}/${CUDA_MINOR}/g'
                            -e 's/\\$$\{pkg:Revision\}/${PKG_REVISION}/g'
                            -e 's/\\$$\{pkg:Timestamp\}/${PKG_TIMESTAMP}/g'
                            -e 's/\\$$\{pkg:Arch\}/${PKG_ARCH}/g'
                            -e 's/\\$$\{pkg:MultiArch\}/${PKG_MULTIARCH}/g'
            ${CMAKE_CURRENT_SOURCE_DIR}/${RPMGEN_IN} > ${RPMPREPDIR}/${RPMGEN}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${RPMGEN_IN}
)

# Build the package
add_custom_target(redhat_build
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PKGDIR}
  COMMAND ${RPMBUILD_EXECUTABLE} --define "_sourcedir ${CMAKE_BINARY_DIR}/pkg/txz"
                                 --define "_rpmdir ${PKGDIR}"
                                 --define "_builddir ${PKGDIR}/build/"
                                 --define "_buildrootdir ${PKGDIR}/buildroot/"
                                 -bb ${CMAKE_BINARY_DIR}/redhat/nccl.spec
  WORKING_DIRECTORY ${RPMPREPDIR}
  DEPENDS redhat_prep nccl_static txz_build
  COMMENT "Building Redhat package with rpmbuild"
)

# Custom target for cleaning redhat package
add_custom_target(redhat_clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${RPMPREPDIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Redhat package build files"
)

# Add redhat targets to the main project
add_dependencies(redhat_build nccl_static txz_build)
