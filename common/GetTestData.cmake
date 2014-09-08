INCLUDE(ExternalProject)

# Set default ExternalProject root directory
set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/data)

ExternalProject_Add(arrayfire_data
    GIT_REPOSITORY git@git.arrayfire.org:arrayfire/arrayfire-data.git
    DOWNLOAD_DIR ${CMAKE_BINARY_DIRECTORY}
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_COMMAND ""
    LOG_DOWNLOAD 0
    LOG_UPDATE 0
    LOG_CONFIGURE 0
    LOG_BUILD 0
    )
