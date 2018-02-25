# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/scan/scan_impl.cu" FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if("${STR}" MATCHES "// SCAN_BINARY_OPS")
        string(REPLACE "// SCAN_BINARY_OPS:" "" TEMP ${STR})
        string(REPLACE " " ";" SCAN_BINARY_OPS ${TEMP})
    endif()
endforeach()

cuda_add_cuda_include_once()

foreach(SCAN_BINARY_OP ${SCAN_BINARY_OPS})
    # When using cuda_compile with older versions of FindCUDA. The generated targets
    # have the same names as the source file. Since we are using the same file for
    # the compilation of these targets we need to rename them before sending them
    # to the cuda_compile command so that it doesn't generate multiple targets with
    # the same name
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/scan/scan_impl.cu"
      DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/scan")
    file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/scan/scan_impl.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/scan/scan_impl_${SCAN_BINARY_OP}.cu")

    cuda_compile(scan_gen_files "${CMAKE_CURRENT_BINARY_DIR}/scan/scan_impl_${SCAN_BINARY_OP}.cu"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first.hpp"
          OPTIONS -DSCAN_BINARY_OP=${SCAN_BINARY_OP} " ${cuda_cxx_flags} -DAFDLL -Xcudafe \"--diag_suppress=1427\""
      )

    message(STATUS "${CMAKE_CURRENT_BINARY_DIR}/scan/scan_impl_${SCAN_BINARY_OP}.cu")

    list(APPEND SCAN_OBJ ${scan_gen_files})
endforeach(SCAN_BINARY_OP ${SCAN_BINARY_OPS})

cuda_compile(scan_nonzero_gen_files "${CMAKE_CURRENT_SOURCE_DIR}/scan/scan_nonzero_impl.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first.hpp"
      OPTIONS -DSCAN_BINARY_OP=${SCAN_BINARY_OP} " ${cuda_cxx_flags} -DAFDLL -Xcudafe \"--diag_suppress=1427\""
  )

cuda_add_library(cuda_scan STATIC ${SCAN_OBJ} ${scan_nonzero_gen_files})
set_target_properties(cuda_scan
  PROPERTIES
    LINKER_LANGUAGE CXX
    FOLDER "Generated Targets"
  )
