# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

find_program(NVPRUNE NAMES nvprune)
cuda_select_nvcc_arch_flags(cuda_architecture_flags ${CUDA_architecture_build_targets})
set(cuda_architecture_flags ${cuda_architecture_flags} CACHE INTERNAL "CUDA compute flags" FORCE)
set(cuda_architecture_flags_readable ${cuda_architecture_flags_readable} CACHE INTERNAL "Readable CUDA compute flags" FORCE)

function(af_detect_and_set_cuda_architectures target)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    string(REGEX REPLACE "sm_([0-9]+)[ ]*" "\\1-real|" cuda_build_targets ${cuda_architecture_flags_readable})
    string(REGEX REPLACE "compute_([0-9]+)[ ]*" "\\1-virtual|" cuda_build_targets ${cuda_build_targets})
    string(REPLACE "|" ";" cuda_build_targets ${cuda_build_targets})

    set_target_properties(${target}
      PROPERTIES
        CUDA_ARCHITECTURES "${cuda_build_targets}")
  else()
    # CMake 3.12 adds deduplication of compile options. This breaks the way the
    # gencode flags are passed into the compiler. these replace instructions add
    # the SHELL: prefix to each of the gencode options so that it is not removed
    # from the command
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
      string(REPLACE ";" "|" cuda_architecture_flags "${cuda_architecture_flags}")
      string(REGEX REPLACE "(-gencode)\\|" "SHELL:\\1 " cuda_architecture_flags2 "${cuda_architecture_flags}")
      string(REPLACE "|" ";" cuda_architecture_flags ${cuda_architecture_flags2})
    endif()
    target_compile_options(${target}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_architecture_flags}>)
  endif()
endfunction()

# The following macro uses a macro defined by
# FindCUDA module from cmake.
function(af_find_static_cuda_libs libname)
  cmake_parse_arguments(fscl "PRUNE" "" "" ${ARGN})

  set(search_name
    "${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  cuda_find_library_local_first(CUDA_${libname}_LIBRARY
    ${search_name} "${libname} static library")

  if(fscl_PRUNE AND AF_WITH_PRUNE_STATIC_CUDA_NUMERIC_LIBS)
    get_filename_component(af_${libname} ${CUDA_${libname}_LIBRARY} NAME)

    set(liboutput ${CMAKE_CURRENT_BINARY_DIR}/${af_${libname}})
    add_custom_command(OUTPUT ${liboutput}.depend
      COMMAND ${NVPRUNE} ${cuda_architecture_flags} ${CUDA_${libname}_LIBRARY} -o ${liboutput}
      COMMAND ${CMAKE_COMMAND} -E touch ${liboutput}.depend
      BYPRODUCTS ${liboutput}
      MAIN_DEPENDENCY ${CUDA_${libname}_LIBRARY}
      COMMENT "Pruning ${CUDA_${libname}_LIBRARY} for ${cuda_build_targets}"
      VERBATIM)
    add_custom_target(prune_${libname}
      DEPENDS ${liboutput}.depend)
    set(cuda_pruned_library_targets ${cuda_pruned_library_targets};prune_${libname} PARENT_SCOPE)

    set(AF_CUDA_${libname}_LIBRARY "${liboutput}" PARENT_SCOPE)
  else()
    set(AF_CUDA_${libname}_LIBRARY ${CUDA_${libname}_LIBRARY} PARENT_SCOPE)
  endif()
  mark_as_advanced(CUDA_${libname}_LIBRARY)
endfunction()

