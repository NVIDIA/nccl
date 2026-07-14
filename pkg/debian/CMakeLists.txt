# Debian package building for NCCL

find_program(DEBUILD_EXECUTABLE debuild)

# Set package directories
set(DEBPREPDIR "${CMAKE_BINARY_DIR}/debian")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/deb")
set(DEBGENFILES libnccl-dev.install libnccl2.install control changelog)

# Get current timestamp in RFC 2822 format
execute_process(COMMAND date -R
                OUTPUT_VARIABLE PKG_TIMESTAMP
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get Debian multiarch tuple
execute_process(COMMAND dpkg-architecture -qDEB_HOST_MULTIARCH
                OUTPUT_VARIABLE PKG_MULTIARCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get Debian architecture
execute_process(COMMAND dpkg-architecture -qDEB_HOST_ARCH
                OUTPUT_VARIABLE PKG_ARCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Custom target for debian package preparation
add_custom_target(debian_prep ALL
  # Create debian prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DEBPREPDIR}
  COMMENT "Preparing Debian package files"
)

# Generate package files from their templates.
foreach(DEBGENFILE ${DEBGENFILES})
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${DEBGENFILE}.in
    ${DEBPREPDIR}/${DEBGENFILE}
    @ONLY
  )
endforeach()

foreach(DEBFILE compat copyright rules)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${DEBFILE}
    ${DEBPREPDIR}/${DEBFILE}
    COPYONLY
  )
endforeach()

# Build the package
add_custom_target(debian_build
  COMMAND cd ${CMAKE_BINARY_DIR} && ${DEBUILD_EXECUTABLE} -eLD_LIBRARY_PATH -uc -us -d -b -Zxz
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PKGDIR}
  COMMAND mv ${CMAKE_SOURCE_DIR}/libnccl*.deb ${PKGDIR}
  WORKING_DIRECTORY ${DEBPREPDIR}
  DEPENDS debian_prep
  COMMENT "Building Debian package with debuild"
)

# Custom target for cleaning debian package
add_custom_target(debian_clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Debian package build files"
)
