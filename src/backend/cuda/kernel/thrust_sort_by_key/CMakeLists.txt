# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu" FILESTRINGS)

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

      # When using cuda_compile with older versions of FindCUDA. The generated targets
      # have the same names as the source file. Since we are using the same file for
      # the compilation of these targets we need to rename them before sending them
      # to the cuda_compile command so that it doesn't generate multiple targets with
      # the same name
      file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/kernel/thrust_sort_by_key")
      file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu"
          "${CMAKE_CURRENT_BINARY_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu")

      cuda_compile(sort_by_key_gen_files
          ${CMAKE_CURRENT_BINARY_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key_impl.hpp
          OPTIONS
            -DSBK_TYPE=${SBK_TYPE}
            -DINSTANTIATESBK_INST=INSTANTIATE${SBK_INST}
            "${platform_flags} ${cuda_cxx_flags} -DAFDLL"
      )

      list(APPEND SORT_OBJ ${sort_by_key_gen_files})
    endforeach(SBK_INST ${SBK_INSTS})
endforeach(SBK_TYPE ${SBK_TYPES})

cuda_add_library(cuda_thrust_sort_by_key STATIC ${SORT_OBJ})

set_target_properties(cuda_thrust_sort_by_key
  PROPERTIES
    LINKER_LANGUAGE CXX
    FOLDER "Generated Targets"
  )
