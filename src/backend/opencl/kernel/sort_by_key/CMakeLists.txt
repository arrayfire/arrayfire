FILE(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp" FILESTRINGS)

FOREACH(STR ${FILESTRINGS})
    IF(${STR} MATCHES "// SBK_TYPES")
        STRING(REPLACE "// SBK_TYPES:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_TYPES ${TEMP})
    ENDIF()
ENDFOREACH()

FOREACH(SBK_TYPE ${SBK_TYPES})
    ADD_LIBRARY(opencl_sort_by_key_${SBK_TYPE} OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key_impl.hpp")
    ADD_DEPENDENCIES(opencl_sort_by_key_${SBK_TYPE} ${cl_kernel_targets} cl2hpp)
    IF(FORGE_FOUND AND NOT USE_SYSTEM_FORGE)
        ADD_DEPENDENCIES(opencl_sort_by_key_${SBK_TYPE} forge)
    ENDIF()
    SET_TARGET_PROPERTIES(opencl_sort_by_key_${SBK_TYPE} PROPERTIES COMPILE_FLAGS "-DTYPE=${SBK_TYPE}")
    LIST(APPEND SORT_BY_KEY_OBJECTS $<TARGET_OBJECTS:opencl_sort_by_key_${SBK_TYPE}>)
ENDFOREACH(SBK_TYPE ${SBK_TYPES})
