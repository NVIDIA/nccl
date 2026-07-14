# Redhat package building for NCCL

find_program(RPMBUILD_EXECUTABLE rpmbuild)

# Set package directories
set(RPMPREPDIR "${CMAKE_BINARY_DIR}/redhat")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/rpm")
set(RPMGEN_IN nccl.spec.in)
set(RPMGEN nccl.spec)

# Get architecture
set(PKG_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Custom target for redhat package preparation
add_custom_target(redhat_prep ALL
  # Create redhat prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RPMPREPDIR}
  COMMENT "Preparing Redhat package files"
)

# Generate the spec file from its template.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/${RPMGEN_IN}
  ${RPMPREPDIR}/${RPMGEN}
  @ONLY
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
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Redhat package build files"
)

# Add redhat targets to the main project
add_dependencies(redhat_build nccl_static txz_build)
