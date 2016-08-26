FILE(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu.in" FILESTRINGS)

FOREACH(STR ${FILESTRINGS})
    IF(${STR} MATCHES "// SBK_TYPES")
        STRING(REPLACE "// SBK_TYPES:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_TYPES ${TEMP})
    ELSEIF(${STR} MATCHES "// SBK_INSTS:")
        STRING(REPLACE "// SBK_INSTS:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_INSTS ${TEMP})
    ENDIF()
ENDFOREACH()

FOREACH(SBK_TYPE ${SBK_TYPES})
    FOREACH(SBK_INST ${SBK_INSTS})
        CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu.in"
        "${CMAKE_CURRENT_BINARY_DIR}/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu")
        ADD_CUSTOM_COMMAND(
          OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu"
          COMMAND  ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key_impl.hpp")
    ENDFOREACH(SBK_INST ${SBK_INSTS})
ENDFOREACH(SBK_TYPE ${SBK_TYPES})

FILE(GLOB thrust_sort_by_key_sources
  "${CMAKE_CURRENT_BINARY_DIR}/thrust_sort_by_key/*.cu"
)

LIST(SORT thrust_sort_by_key_sources)
