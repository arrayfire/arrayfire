# Tailored after https://github.com/GerbilSoft/mcrecover/blob/master/cmake/macros/SplitDebugInformation.cmake
# Minor modifications to original

if (NOT WIN32)
  include(CMakeFindBinUtils)
  if (NOT APPLE AND NOT CMAKE_OBJCOPY)
    message("'objcopy' tool not found; debug information will not be split.")
  elseif (NOT CMAKE_STRIP)
    message("'strip' tool not found; debug information will not be split.")
  elseif (APPLE)
    # TODO(pradeep) debug info splits on OSX are disabled
    # this section of elseif will be removed when Apple support is added
    message("Debug information is not split on OSX")
  endif ()
endif (NOT WIN32)

function(af_split_debug_info _target _destination_dir)
  set(SPLIT_TOOL_EXISTS ON)
  if (WIN32)
    set(SPLIT_TOOL_EXISTS OFF)
    if (MSVC)
      install(FILES
        $<TARGET_PDB_FILE:${_target}>
        DESTINATION ${_destination_dir}
        COMPONENT "${_target}_debug_symbols"
        )
    endif()
  elseif (NOT APPLE AND NOT CMAKE_OBJCOPY)
    set(SPLIT_TOOL_EXISTS OFF)
  elseif (NOT CMAKE_STRIP)
    set(SPLIT_TOOL_EXISTS OFF)
  elseif (APPLE)
    # TODO(pradeep) debug info splits on OSX are disabled
    # this section of elseif will be removed when Apple support is added
    set(SPLIT_TOOL_EXISTS OFF)
  endif ()

  if (SPLIT_TOOL_EXISTS)
    get_target_property(TARGET_TYPE ${_target} TYPE)
    set(PREFIX_EXPR_1
      "$<$<STREQUAL:$<TARGET_PROPERTY:${_target},PREFIX>,>:${CMAKE_${TARGET_TYPE}_PREFIX}>")
    set(PREFIX_EXPR_2
      "$<$<NOT:$<STREQUAL:$<TARGET_PROPERTY:${_target},PREFIX>,>>:$<TARGET_PROPERTY:${_target},PREFIX>>")
    set(PREFIX_EXPR_FULL "${PREFIX_EXPR_1}${PREFIX_EXPR_2}")

    # If a custom OUTPUT_NAME was specified, use it.
    set(OUTPUT_NAME_EXPR_1
        "$<$<STREQUAL:$<TARGET_PROPERTY:${_target},OUTPUT_NAME>,>:${_target}>")
    set(OUTPUT_NAME_EXPR_2
        "$<$<NOT:$<STREQUAL:$<TARGET_PROPERTY:${_target},OUTPUT_NAME>,>>:$<TARGET_PROPERTY:${_target},OUTPUT_NAME>>")
    set(OUTPUT_NAME_EXPR "${OUTPUT_NAME_EXPR_1}${OUTPUT_NAME_EXPR_2}")
    set(OUTPUT_NAME_FULL "${PREFIX_EXPR_FULL}${OUTPUT_NAME_EXPR}$<TARGET_PROPERTY:${_target},POSTFIX>")

    set(SPLIT_DEBUG_TARGET_EXT ".debug")
    if(APPLE)
        set(SPLIT_DEBUG_TARGET_EXT ".dSYM")
    endif()
    set(SPLIT_DEBUG_SOURCE "$<TARGET_FILE:${_target}>")
    set(SPLIT_DEBUG_TARGET_NAME
        "$<TARGET_FILE_DIR:${_target}>/${OUTPUT_NAME_FULL}")
    set(SPLIT_DEBUG_TARGET
        "${SPLIT_DEBUG_TARGET_NAME}${SPLIT_DEBUG_TARGET_EXT}")

    if(APPLE)
      add_custom_command(TARGET ${_target} POST_BUILD
          COMMAND dsymutil ${SPLIT_DEBUG_SOURCE} -o ${SPLIT_DEBUG_TARGET}
          #TODO(pradeep) From initial research stripping debug info from
          # is removing debug LC_ID_DYLIB command also which is make
          # shared library unusable. Confirm this from OSX expert
          # and remove these comments and below command
          #COMMAND ${CMAKE_STRIP} --strip-debug ${SPLIT_DEBUG_SOURCE}
        )
    else(APPLE)
      add_custom_command(TARGET ${_target} POST_BUILD
        COMMAND ${CMAKE_OBJCOPY}
          --only-keep-debug ${SPLIT_DEBUG_SOURCE} ${SPLIT_DEBUG_TARGET}
        COMMAND ${CMAKE_STRIP}
          --strip-debug ${SPLIT_DEBUG_SOURCE}
        COMMAND ${CMAKE_OBJCOPY}
          --add-gnu-debuglink=${SPLIT_DEBUG_TARGET} ${SPLIT_DEBUG_SOURCE}
        )
    endif()

    install(FILES
      ${SPLIT_DEBUG_TARGET}
      DESTINATION ${_destination_dir}
      COMPONENT "${OUTPUT_NAME_FULL}_debug_symbols"
      )

    # Make sure the file is deleted on `make clean`.
    set_property(DIRECTORY APPEND
      PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${SPLIT_DEBUG_TARGET})
  endif(SPLIT_TOOL_EXISTS)
endfunction(af_split_debug_info)
