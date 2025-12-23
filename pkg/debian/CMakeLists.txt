# Debian package building for NCCL

find_program(DEBUILD_EXECUTABLE debuild)
find_program(SED_EXECUTABLE sed)

# Set package directories
set(DEBPREPDIR "${CMAKE_BINARY_DIR}/debian")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/deb")
set(DEBGENFILES libnccl-dev.install libnccl2.install control changelog)
set(DEBFILES compat copyright rules $(DEBGENFILES))

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

# for loop to generate the files
foreach(DEBGENFILE ${DEBGENFILES})
  add_custom_command(
    TARGET debian_prep
    COMMAND ${SED_EXECUTABLE} -e 's/\\$$\{nccl:Major\}/${NCCL_MAJOR}/g'
                              -e 's/\\$$\{nccl:Minor\}/${NCCL_MINOR}/g'
                              -e 's/\\$$\{nccl:Patch\}/${NCCL_PATCH}/g'
                              -e 's/\\$$\{nccl:Suffix\}/${NCCL_SUFFIX}/g'
                              -e 's/\\$$\{pkg:Revision\}/${PKG_REVISION}/g'
                              -e 's/\\$$\{cuda:Major\}/${CUDA_MAJOR}/g'
                              -e 's/\\$$\{cuda:Minor\}/${CUDA_MINOR}/g'
                              -e 's/\\$$\{pkg:Timestamp\}/${PKG_TIMESTAMP}/g'
                              -e 's/\\$$\{pkg:MultiArch\}/${PKG_MULTIARCH}/g'
                              -e 's/\\$$\{pkg:Arch\}/${PKG_ARCH}/g'
              ${CMAKE_CURRENT_SOURCE_DIR}/${DEBGENFILE}.in > ${DEBPREPDIR}/${DEBGENFILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${DEBGENFILE}.in
)
endforeach()

foreach(DEBFILE ${DEBFILES})
  add_custom_command(TARGET debian_prep
                     COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_SOURCE_DIR}/${DEBFILE} ${DEBPREPDIR}/${DEBFILE}
                     DEPENDS ${DEBFILE})
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
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${DEBPREPDIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Debian package build files"
)
