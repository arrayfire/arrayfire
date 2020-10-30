add_library(glad_interface INTERFACE)

target_include_directories(glad_interface
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_sources(glad_interface
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/glad.c>
)

target_link_libraries(glad_interface
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_DL_LIBS}>
)
