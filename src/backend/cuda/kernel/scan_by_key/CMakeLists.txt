# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cu" FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_BINARY_OPS")
        string(REPLACE "// SBK_BINARY_OPS:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_BINARY_OPS ${TEMP})
    endif()
endforeach()

cuda_add_cuda_include_once()

foreach(SBK_BINARY_OP ${SBK_BINARY_OPS})
    # When using cuda_compile with older versions of FindCUDA. The generated targets
    # have the same names as the source file. Since we are using the same file for
    # the compilation of these targets we need to rename them before sending them
    # to the cuda_compile command so that it doesn't generate multiple targets with
    # the same name
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cu"
      DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/kernel/scan_by_key")
    file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/kernel/scan_by_key/scan_by_key_impl.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cu")

    cuda_compile(scan_by_key_gen_files "${CMAKE_CURRENT_BINARY_DIR}/kernel/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cu"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim_by_key_impl.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first_by_key_impl.hpp"
          OPTIONS -DSBK_BINARY_OP=${SBK_BINARY_OP} "${cuda_cxx_flags} -DAFDLL"
      )

    list(APPEND SCAN_OBJ ${scan_by_key_gen_files})
endforeach(SBK_BINARY_OP ${SBK_BINARY_OPS})

cuda_add_library(cuda_scan_by_key STATIC ${SCAN_OBJ})
set_target_properties(cuda_scan_by_key
  PROPERTIES
    LINKER_LANGUAGE CXX
    FOLDER "Generated Targets"
  )
