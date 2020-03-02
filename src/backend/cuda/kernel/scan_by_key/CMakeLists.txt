# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cpp" FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_BINARY_OPS")
        string(REPLACE "// SBK_BINARY_OPS:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_BINARY_OPS ${TEMP})
    endif()
endforeach()

foreach(SBK_BINARY_OP ${SBK_BINARY_OPS})
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cpp"
  )

  list(
    APPEND
    scan_by_key_sources
    "${CMAKE_CURRENT_BINARY_DIR}/kernel/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cpp"
  )
endforeach(SBK_BINARY_OP ${SBK_BINARY_OPS})
