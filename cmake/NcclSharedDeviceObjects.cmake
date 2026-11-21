# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
# Device objects for the libnccl.so link.
#
# The device objects are compiled with -rdc. Each one carries its relocatable cubins in __nv_relfatbin,
# the input of the device link. When the objects are also given to the host linker, all of these
# sections are copied into libnccl.so, where only load-time JIT linking could use the relocatable
# cubins. The objects are added to libnccl.so as before, so CMake's device link is unchanged; a linker
# launcher (src/device/nvprune_link.sh) only replaces them, in the host link command, with copies
# without the relocatable cubins (src/device/nvprune_copy.sh, as in the Makefile build), or with the
# originals where that cannot be done safely. libnccl_static.a keeps the original objects.
#
# Not used with NCCL_PRUNE_SHARED_RELOCATABLE=OFF, with a multi-config generator, with CUDA older than
# 11.3 (first nvprune with --no-relocatable-elf), when the target already has a CXX linker launcher,
# or when nvprune, cuobjdump or readelf is missing.

option(NCCL_PRUNE_SHARED_RELOCATABLE "Link libnccl.so from device objects without pre-link relocatable cubins" ON)

# Add the objects of the given OBJECT libraries to the shared library target.
function(nccl_add_shared_device_objects target)
    set(object_libs ${ARGN})
    set(lines "")
    foreach(lib IN LISTS object_libs)
        target_sources(${target} PRIVATE $<TARGET_OBJECTS:${lib}>)
        string(APPEND lines "$<JOIN:$<TARGET_OBJECTS:${lib}>,\n>\n")
    endforeach()
    set(reason "")
    get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    get_target_property(launcher ${target} CXX_LINKER_LAUNCHER)
    if(NOT NCCL_PRUNE_SHARED_RELOCATABLE)
        set(reason "NCCL_PRUNE_SHARED_RELOCATABLE is OFF")
    elseif(multi_config)
        set(reason "multi-config generator")
    elseif(CUDAToolkit_VERSION VERSION_LESS 11.3)
        set(reason "CUDA ${CUDAToolkit_VERSION} is older than 11.3")
    elseif(launcher)
        set(reason "CXX_LINKER_LAUNCHER is already set")
    else()
        # From the toolkit of the CUDA compiler only.
        find_program(NCCL_NVPRUNE nvprune PATHS "${CUDAToolkit_BIN_DIR}" NO_DEFAULT_PATH)
        find_program(NCCL_CUOBJDUMP cuobjdump PATHS "${CUDAToolkit_BIN_DIR}" NO_DEFAULT_PATH)
        if(NOT CMAKE_READELF)
            find_program(CMAKE_READELF readelf)
        endif()
        if(NOT NCCL_NVPRUNE OR NOT NCCL_CUOBJDUMP OR NOT CMAKE_READELF)
            set(reason "nvprune, cuobjdump or readelf not found")
        endif()
    endif()
    if(reason)
        message(STATUS "libnccl.so: linking the original device objects (${reason})")
        return()
    endif()
    message(STATUS "libnccl.so: linking device objects without pre-link relocatable cubins")

    # The launcher runs only in the host link. It remakes a copy when the copy is missing or older than
    # its object, the device link object or the scripts.
    set(dir "${CMAKE_CURRENT_BINARY_DIR}/${target}_shared_device")
    file(GENERATE OUTPUT "${dir}/objects.txt" CONTENT "${lines}")
    set_target_properties(${target} PROPERTIES CXX_LINKER_LAUNCHER
        "sh;${PROJECT_SOURCE_DIR}/src/device/nvprune_link.sh;${dir};${NCCL_NVPRUNE};${NCCL_CUOBJDUMP};${CMAKE_READELF};--")
endfunction()
