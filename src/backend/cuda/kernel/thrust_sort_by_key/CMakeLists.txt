# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

file(STRINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu"
    FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_TYPES")
        string(REPLACE "// SBK_TYPES:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_TYPES ${TEMP})
    elseif(${STR} MATCHES "// SBK_INSTS:")
        string(REPLACE "// SBK_INSTS:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_INSTS ${TEMP})
    endif()
endforeach()

foreach(SBK_TYPE ${SBK_TYPES})
  foreach(SBK_INST ${SBK_INSTS})
    set(INSTANTIATESBK_INST "INSTANTIATE${SBK_INST}")

    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu"
    )

    list(
      APPEND
      thrust_sort_sources
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu"
    )
  endforeach(SBK_INST ${SBK_INSTS})
endforeach(SBK_TYPE ${SBK_TYPES})
