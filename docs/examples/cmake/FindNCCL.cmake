#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

#[=======================================================================[.rst:
FindNCCL
--------

Finds the NVIDIA Collective Communication Library (NCCL):

.. code-block:: cmake

find_package(NCCL [<version>|<version_range>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``NCCL::nccl``

  Target that encapsulates the NCCL usage requirements.  It is available
  only when NCCL is found

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``NCCL_FOUND``
  Boolean indicating whether (the requested version of) NCCL was found.

``NCCL_VERSION``
  The version of the NCCL library which was found.

``NCCL_INCLUDE_DIRS``
  Include directories needed to use NCCL.

``NCCL_LIBRARIES``
  Libraries needed to link to NCCL.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``NCCL_INCLUDE_DIR``
  The directory containing ``nccl.h``.

``NCCL_LIBRARY``
  The path to the NCCL library.

Hints
^^^^^

``NCCL_ROOT``
  A user may set this variable to a NCCL installation root to help locate
  NCCL in a custom installation path.

#]=======================================================================]

# Capture the find_package version and quiet args to forward
# to the CONFIG and pkgconfig searches.

set(_NCCL_FP_VERSION ${NCCL_FIND_VERSION})
if(NCCL_FIND_VERSION_RANGE)
  set(_NCCL_FP_VERSION ${NCCL_FIND_VERSION_RANGE})
endif()

set(_NCCL_FP_EXACT)
if(NCCL_FIND_VERSION_EXACT)
  set(_NCCL_FP_EXACT EXACT)
endif()

set(_NCCL_FP_QUIET)
if(NCCL_FIND_QUIETLY)
  set(_NCCL_FP_QUIET QUIET)
endif()

# Look for the CMake package first
find_package(NCCL
  ${_NCCL_FP_VERSION} ${_NCCL_FP_EXACT} ${_NCCL_FP_QUIET}
  CONFIG QUIET
)

include(FindPackageHandleStandardArgs)
set(_NCCL_FPHSA_ARGS)
if(NCCL_FOUND AND NCCL_CONFIG)
  list(APPEND _NCCL_FPHSA_ARGS CONFIG_MODE)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    list(APPEND _NCCL_FPHSA_ARGS HANDLE_VERSION_RANGE)
  endif()
  find_package_handle_standard_args(NCCL ${_NCCL_FPHSA_ARGS})

  # Check for older compatibility imported targets
  if(NCCL_FOUND AND NOT TARGET NCCL::nccl AND TARGET nccl::nccl)
    add_library(NCCL::nccl ALIAS nccl::nccl)
  endif()

  # Set the compatibility vars
  if(NOT NCCL_INCLUDE_DIRS)
    if(TARGET NCCL::nccl_headers)
      set(_NCCL_INC_TARGET NCCL::nccl_headers)
    else()
      set(_NCCL_INC_TARGET NCCL::nccl)
    endif()
    get_target_property(
      NCCL_INCLUDE_DIRS
      ${_NCCL_INC_TARGET}
      INTERFACE_INCLUDE_DIRECTORIES
    )
    unset(_NCCL_INC_TARGET)
  endif()
  if(NOT NCCL_LIBRARIES)
    set(NCCL_LIBRARIES NCCL::nccl)
  endif()
else()
  # Clear state from a half found / rejected CONFIG search
  unset(NCCL_FOUND)
  unset(NCCL_CONFIG)
  unset(NCCL_VERSION)

  # NCCLConfig.cmake not found, try manual search seeded by pkgconfig
  find_package(CUDAToolkit ${_NCCL_FP_QUIET})

  if(CUDAToolkit_FOUND)
    set(_NCCL_INCLUDE_PATHS)
    set(_NCCL_LIBRARY_PATHS)
    set(_NCCL_LIBRARY_NAMES)

    # Use pkgconfig to guide the manual search and discovery
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      # build the pkgconfig module spec
      set(_PC_NCCL_SPEC "nccl")

      if(NCCL_FIND_VERSION)
        if(_NCCL_FP_EXACT)
          string(APPEND _PC_NCCL_SPEC
            " = ${NCCL_FIND_VERSION}"
          )
        else()
          string(APPEND _PC_NCCL_SPEC
            " >= ${NCCL_FIND_VERSION}"
          )
          if(NCCL_FIND_VERSION_RANGE)
            if(NCCL_FIND_VERSION_RANGE_MAX STREQUAL "INCLUDE")
              list(APPEND _PC_NCCL_SPEC
                "nccl <= ${NCCL_FIND_VERSION_MAX}"
              )
            else()
              list(APPEND _PC_NCCL_SPEC
                "nccl < ${NCCL_FIND_VERSION_MAX}"
              )
            endif()
          endif()
        endif()
      endif()

      pkg_check_modules(PC_NCCL ${_NCCL_FP_QUIET} ${_PC_NCCL_SPEC})

      # Populate hints from pkg_config
      if(PC_NCCL_FOUND)
        if(PC_NCCL_INCLUDE_DIRS)
          list(APPEND _NCCL_INCLUDE_PATHS
            PATHS ${PC_NCCL_INCLUDE_DIRS}
            NO_DEFAULT_PATH
          )
        endif()
        if(PC_NCCL_LIBRARY_DIRS)
          list(APPEND _NCCL_LIBRARY_PATHS
            PATHS ${PC_NCCL_LIBRARY_DIRS}
            NO_DEFAULT_PATH
          )
        endif()
        if(PC_NCCL_LIBRARIES)
          list(APPEND _NCCL_LIBRARY_NAMES NAMES ${PC_NCCL_LIBRARIES})
        endif()
      endif()
      unset(_PC_NCCL_SPEC)
    endif()

    if(NOT _NCCL_LIBRARY_NAMES)
      list(APPEND _NCCL_LIBRARY_NAMES NAMES nccl)

      # Handle the special case for the nvidia-nccl python wheels on
      # Linux that don't ship the unversioned symlink
      # libnccl.so -> libnccl.so.2
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        list(APPEND _NCCL_LIBRARY_NAMES libnccl.so.2)
      endif()
    endif()

    # Perform the manual search, possibly guided by pkgconfig
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h ${_NCCL_INCLUDE_PATHS})
    find_library(NCCL_LIBRARY ${_NCCL_LIBRARY_NAMES} ${_NCCL_LIBRARY_PATHS})
    mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
    unset(_NCCL_INCLUDE_PATHS)
    unset(_NCCL_LIBRARY_PATHS)
    unset(_NCCL_LIBRARY_NAMES)

    # Parse NCCL_VERSION from nccl.h
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY AND NOT NCCL_VERSION)
      set(_NCCL_VERSION_REGEX
        [[^#define[ \t]+NCCL_(MAJOR|MINOR|PATCH)[ \t]+([0-9]+)[ \t]*$]]
      )
      file(STRINGS
        "${NCCL_INCLUDE_DIR}/nccl.h"
        _NCCL_HEADER_LINES
        REGEX ${_NCCL_VERSION_REGEX}
      )
      if(_NCCL_HEADER_LINES)
        list(TRANSFORM _NCCL_HEADER_LINES
          REPLACE ${_NCCL_VERSION_REGEX} [[\2]]
        )
        list(JOIN _NCCL_HEADER_LINES "." NCCL_VERSION)
      endif()
      unset(_NCCL_VERSION_REGEX)
      unset(_NCCL_HEADER_LINES)
    endif()

    # Fallback to PC_NCCL_VERSION if the header couldn't be parsed and
    # is found in pkgconfig's location
    if(NOT NCCL_VERSION AND PC_NCCL_VERSION AND
      PC_NCCL_INCLUDE_DIRS AND NCCL_INCLUDE_DIR IN_LIST PC_NCCL_INCLUDE_DIRS)
      set(NCCL_VERSION ${PC_NCCL_VERSION})
    endif()
  endif()

  list(APPEND _NCCL_FPHSA_ARGS
    REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY CUDAToolkit_FOUND
    VERSION_VAR NCCL_VERSION
  )
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    list(APPEND _NCCL_FPHSA_ARGS HANDLE_VERSION_RANGE)
  endif()
  find_package_handle_standard_args(NCCL ${_NCCL_FPHSA_ARGS})

  # Create the imported targets
  if(NCCL_FOUND)
    if(NOT TARGET NCCL::nccl)
      add_library(NCCL::nccl UNKNOWN IMPORTED)
      set_target_properties(NCCL::nccl PROPERTIES
        IMPORTED_LOCATION "${NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
      )
      if(TARGET CUDA::toolkit)
        set_target_properties(NCCL::nccl PROPERTIES
          INTERFACE_LINK_LIBRARIES CUDA::toolkit
        )
      endif()
    endif()

    # Set the compatibility variables
    set(NCCL_INCLUDE_DIRS "${NCCL_INCLUDE_DIR}")
    set(NCCL_LIBRARIES "${NCCL_LIBRARY}")
  endif()
endif()

# Cleanup after ourselves
unset(_NCCL_FPHSA_ARGS)
unset(_NCCL_FP_QUIET)
unset(_NCCL_FP_EXACT)
unset(_NCCL_FP_VERSION)
