FILE(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cu.in" FILESTRINGS)

FOREACH(STR ${FILESTRINGS})
    IF(${STR} MATCHES "// SBK_DIMS")
        STRING(REPLACE "// SBK_DIMS:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_DIMS ${TEMP})
    ELSEIF(${STR} MATCHES "// SBK_BINARY_OPS")
        STRING(REPLACE "// SBK_BINARY_OPS:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_BINARY_OPS ${TEMP})
    ENDIF()
ENDFOREACH()

FOREACH(SBK_DIM ${SBK_DIMS})
    STRING(TOUPPER ${SBK_DIM} SBK_DIM_UPPER)
    FOREACH(SBK_BINARY_OP ${SBK_BINARY_OPS})
        CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cu.in"
          "${CMAKE_CURRENT_BINARY_DIR}/scan_by_key/scan_${SBK_DIM}_by_key_impl_${SBK_BINARY_OP}.cu")
        ADD_CUSTOM_COMMAND(
          OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/scan_by_key/scan_${SBK_DIM}_by_key_impl_${SBK_BINARY_OP}.cu"
          COMMAND  ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/scan_by_key/scan_${SBK_DIM}_by_key_impl_${SBK_BINARY_OP}.cu"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_${SBK_DIM}_by_key_impl.hpp")
    ENDFOREACH(SBK_BINARY_OP ${SBK_BINARY_OPS})
ENDFOREACH(SBK_DIM ${SBK_DIMS})

FILE(GLOB scan_by_key_sources
  "${CMAKE_CURRENT_BINARY_DIR}/scan_by_key/*.cu"
)

LIST(SORT scan_by_key_sources)
