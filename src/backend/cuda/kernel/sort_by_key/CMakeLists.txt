FILE(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cu.in" FILESTRINGS)

FOREACH(STR ${FILESTRINGS})
    IF(${STR} MATCHES "// SBK_TYPES")
        STRING(REPLACE "// SBK_TYPES:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_TYPES ${TEMP})
    ELSEIF(${STR} MATCHES "// SBK_DIRS:")
        STRING(REPLACE "// SBK_DIRS:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_DIRS ${TEMP})
    ELSEIF(${STR} MATCHES "// SBK_INSTS:")
        STRING(REPLACE "// SBK_INSTS:" "" TEMP ${STR})
        STRING(REPLACE " " ";" SBK_INSTS ${TEMP})
    ENDIF()
ENDFOREACH()

FOREACH(SBK_TYPE ${SBK_TYPES})
    FOREACH(SBK_DIR ${SBK_DIRS})
        FOREACH(SBK_INST ${SBK_INSTS})
            CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cu.in"
            "${CMAKE_CURRENT_BINARY_DIR}/sort_by_key/sort_by_key_impl_${SBK_TYPE}_${SBK_DIR}_${SBK_INST}.cu")
        ENDFOREACH(SBK_INST ${SBK_INSTS})
    ENDFOREACH(SBK_DIR ${SBK_DIRS})
ENDFOREACH(SBK_TYPE ${SBK_TYPES})

FILE(GLOB sort_by_key_sources
  "${CMAKE_CURRENT_BINARY_DIR}/sort_by_key/*.cu"
)

LIST(SORT sort_by_key_sources)
