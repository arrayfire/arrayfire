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
        $<$<CONFIG:Debug>:$<TARGET_PDB_FILE:${_target}>>
        $<$<CONFIG:RelWithDebInfo>:$<TARGET_PDB_FILE:${_target}>>
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
    get_target_property(TRGT_PREFIX ${_target} PREFIX)
    if(TRGT_PREFIX)
      set(prefix ${TRGT_PREFIX})
    else()
      get_target_property(TRGT_TYPE ${_target} TYPE)
      set(prefix "${CMAKE_${TRGT_TYPE}_PREFIX}")
    endif()

    get_target_property(TRGT_OUT_NAME ${_target} OUTPUT_NAME)
    if(TRGT_OUT_NAME)
      set(outName ${TRGT_OUT_NAME})
    else()
      set(outName "${_target}")
    endif()

    get_target_property(TRGT_POSTFIX ${_target} POSTFIX)
    if(TRGT_POSTFIX)
      set(postfix ${TRGT_POSTFIX})
    else()
      get_target_property(TRGT_TYPE ${_target} TYPE)
      set(postfix "${CMAKE_${TRGT_TYPE}_POSTFIX}")
    endif()

    set(OUT_NAME "${prefix}${outName}")
    set(OUT_NAME_WE "${OUT_NAME}${postfix}")
    set(SPLIT_DEBUG_OUT_FILE_EXT ".debug")
    if(APPLE)
      set(SPLIT_DEBUG_OUT_FILE_EXT ".dSYM")
    endif()
    set(SPLIT_DEBUG_SRC_FILE "$<TARGET_FILE:${_target}>")
    set(SPLIT_DEBUG_OUT_NAME "$<TARGET_FILE_DIR:${_target}>/${OUT_NAME_WE}")
    set(SPLIT_DEBUG_OUT_FILE "${SPLIT_DEBUG_OUT_NAME}${SPLIT_DEBUG_OUT_FILE_EXT}")

    if(APPLE)
      add_custom_command(TARGET ${_target} POST_BUILD
          COMMAND dsymutil ${SPLIT_DEBUG_SRC_FILE} -o ${SPLIT_DEBUG_OUT_FILE}
          #TODO(pradeep) From initial research stripping debug info from
          # is removing debug LC_ID_DYLIB command also which is make
          # shared library unusable. Confirm this from OSX expert
          # and remove these comments and below command
          #COMMAND ${CMAKE_STRIP} --strip-debug ${SPLIT_DEBUG_SRC_FILE}
        )
    else(APPLE)
      add_custom_command(TARGET ${_target} POST_BUILD
        COMMAND ${CMAKE_OBJCOPY}
          --only-keep-debug ${SPLIT_DEBUG_SRC_FILE} ${SPLIT_DEBUG_OUT_FILE}
        COMMAND ${CMAKE_STRIP}
          --strip-debug ${SPLIT_DEBUG_SRC_FILE}
        COMMAND ${CMAKE_OBJCOPY}
          --add-gnu-debuglink=${SPLIT_DEBUG_OUT_FILE} ${SPLIT_DEBUG_SRC_FILE}
        )
    endif()

    install(FILES ${SPLIT_DEBUG_OUT_FILE}
      DESTINATION ${_destination_dir}
      COMPONENT "${OUT_NAME}_debug_symbols"
      )

    # Make sure the file is deleted on `make clean`.
    set_property(DIRECTORY APPEND
      PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${SPLIT_DEBUG_OUT_FILE})
  endif(SPLIT_TOOL_EXISTS)
endfunction(af_split_debug_info)
