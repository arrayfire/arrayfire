generate_product_version(af_unified_ver_res_file
  FILE_NAME "af"
  FILE_DESCRIPTION "Unified Backend Dynamic-link library"
)

add_library(af "")
add_library(ArrayFire::af ALIAS af)

target_sources(af
  PRIVATE
    ${af_unified_ver_res_file}
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/arith.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/blas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_test_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lapack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ml.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moments.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/signal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/symbol_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/symbol_manager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vision.cpp
  )

if(OpenCL_FOUND)
  target_sources(af
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/opencl.cpp
  )

  target_include_directories(af
    SYSTEM PRIVATE
      $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>)

endif()

if(CUDA_FOUND)
  target_sources(af
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp)

  target_include_directories(af
    SYSTEM PRIVATE
      ${CUDA_INCLUDE_DIRS})
endif()

target_sources(af
  PRIVATE
    ${ArrayFire_SOURCE_DIR}/src/api/c/type_util.cpp
    ${ArrayFire_SOURCE_DIR}/src/api/c/version.cpp
    ${ArrayFire_SOURCE_DIR}/src/backend/common/Logger.cpp
    ${ArrayFire_SOURCE_DIR}/src/backend/common/Logger.hpp
    ${ArrayFire_SOURCE_DIR}/src/backend/common/constants.cpp
    ${ArrayFire_SOURCE_DIR}/src/backend/common/dim4.cpp
    ${ArrayFire_SOURCE_DIR}/src/backend/common/err_common.cpp
    ${ArrayFire_SOURCE_DIR}/src/backend/common/util.cpp
    ${ArrayFire_SOURCE_DIR}/src/backend/common/util.hpp
    ${ArrayFire_SOURCE_DIR}/src/backend/common/deprecated.hpp
  )

arrayfire_set_default_cxx_flags(af)
if(WIN32)
  target_sources(af
    PRIVATE
      ${ArrayFire_SOURCE_DIR}/src/backend/common/module_loading_windows.cpp)
else()
  target_sources(af
    PRIVATE
      ${ArrayFire_SOURCE_DIR}/src/backend/common/module_loading_unix.cpp)
endif()

target_compile_definitions(af PRIVATE AF_UNIFIED)

target_include_directories(af
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${ArrayFire_SOURCE_DIR}/src/api/c
    ${ArrayFire_SOURCE_DIR}/src/api/unified)

target_include_directories(af
  SYSTEM PRIVATE
    $<TARGET_PROPERTY:afcommon_interface,INTERFACE_INCLUDE_DIRECTORIES>
    ${CMAKE_BINARY_DIR}
  )

target_link_libraries(af
  PRIVATE
    af_spdlog
    cpp_api_interface
    Threads::Threads
    Boost::boost
    ${CMAKE_DL_LIBS}
  )

if(TARGET fmt::fmt)
  target_link_libraries(af
    PRIVATE
      fmt::fmt
  )
endif()

install(TARGETS af
  EXPORT ArrayFireUnifiedTargets
  COMPONENT unified
  PUBLIC_HEADER DESTINATION af
  RUNTIME DESTINATION ${AF_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${AF_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${AF_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${AF_INSTALL_INC_DIR}
  )

af_split_debug_info(af ${AF_INSTALL_LIB_DIR})

# install(TARGETS af EXPORT AF DESTINATION "${AF_INSTALL_LIB_DIR}"
#   COMPONENT libraries)
#
# if(APPLE)
#   INSTALL(SCRIPT "${PROJECT_SOURCE_DIR}/CMakeModules/osx_install/InstallTool.cmake")
# endif(APPLE)
#
# export(TARGETS af FILE ArrayFireUnified.cmake)
# install(EXPORT AF DESTINATION "${AF_INSTALL_CMAKE_DIR}"
#   COMPONENT cmake
#   FILE ArrayFireUnified.cmake)

source_group(include REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/include/*)
source_group(source REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/*|${ArrayFire_SOURCE_DIR}/src/backend/common/*)
source_group(api\\cpp REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/c/*)
source_group("" FILES CMakeLists.txt)
