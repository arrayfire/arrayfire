# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause


# The following macro uses a macro defined by
# FindCUDA module from cmake.
function(af_find_static_cuda_libs libname)
  set(search_name
    "${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  cuda_find_library_local_first(CUDA_${libname}_LIBRARY
    ${search_name} "${libname} static library")
  mark_as_advanced(CUDA_${libname}_LIBRARY)
endfunction()

## Copied from FindCUDA.cmake
## The target_link_library needs to link with the cuda libraries using
## PRIVATE
function(cuda_add_library cuda_target)
  cuda_add_cuda_include_once()

  # Separate the sources from the options
  cuda_get_sources_and_options(_sources _cmake_options _options ${ARGN})
  cuda_build_shared_library(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  cuda_wrap_srcs( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  cuda_compute_separable_compilation_object_file_name(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  cuda_link_separable_compilation_objects("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${cuda_target}
      PRIVATE ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    POSITION_INDEPENDENT_CODE ON
  )
endfunction()
