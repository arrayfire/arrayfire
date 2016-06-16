FILE(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cu.in" FILESTRINGS)

FOREACH(STR ${FILESTRINGS})
    IF(${STR} MATCHES "// SBK_BINARY_OPS")
        STRING(REPLACE "// SBK_BINARY_OPS:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_BINARY_OPS ${TEMP})
    ENDIF()
ENDFOREACH()

FOREACH(SBK_BINARY_OP ${SBK_BINARY_OPS})
    CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cu.in"
        "${CMAKE_CURRENT_BINARY_DIR}/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cu")
    ADD_CUSTOM_COMMAND(
        OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cu"
        COMMAND  ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cu"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first_by_key_impl.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim_by_key_impl.hpp")
ENDFOREACH(SBK_BINARY_OP ${SBK_BINARY_OPS})

FILE(GLOB scan_by_key_sources
  "${CMAKE_CURRENT_BINARY_DIR}/scan_by_key/*.cu"
)

LIST(SORT scan_by_key_sources)
