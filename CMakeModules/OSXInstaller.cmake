#
# Builds ArrayFire Installers for OSX
#
include(CMakeParseArguments)
include(${CMAKE_MODULE_PATH}/Version.cmake)

set(BIN2CPP_PROGRAM "bin2cpp")

function(PKG_BUILD)
    cmake_parse_arguments(ARGS "" "INSTALL_LOCATION;IDENTIFIER;PATH_TO_FILES;PKG_NAME" "FILTERS" ${ARGN})

    foreach(filter ${ARGS_FILTERS})
        LIST(APPEND  FILTER_LIST --filter ${filter})
    endforeach()

    EXECUTE_PROCESS(COMMAND pkgbuild    --install-location  ${ARGS_INSTALL_LOCATION}
                                        --identifier        ${ARGS_IDENTIFIER}
                                        --root ${ARGS_PATH_TO_FILES}
                                        ${FILTER_LIST}
                                        ${ARGS_PKG_NAME}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction(PKG_BUILD)

function(PRODUCT_BUILD)
    SET(DISTRIBUTION_FILE       "${CMAKE_MODULE_PATH}/distribution.dist")
    SET(DISTRIBUTION_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/distribution.dist.out")

    SET(WELCOME_FILE       "${CMAKE_MODULE_PATH}/welcome.html")
    SET(WELCOME_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/welcome.html.out")

    SET(README_FILE       "${CMAKE_MODULE_PATH}/readme.html")
    SET(README_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/readme.html.out")

    SET(AF_TITLE    "ArrayFire ${AF_VERSION}")
    CONFIGURE_FILE(${DISTRIBUTION_FILE} ${DISTRIBUTION_FILE_OUT})
    CONFIGURE_FILE(${WELCOME_FILE} ${WELCOME_FILE_OUT})
    CONFIGURE_FILE(${README_FILE} ${README_FILE_OUT})

    SET(PACKAGE_NAME "Install ArrayFire.pkg")
    EXECUTE_PROCESS(COMMAND productbuild    --distribution  ${DISTRIBUTION_FILE_OUT}
                                            ${PACKAGE_NAME}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction(PRODUCT_BUILD)


function(OSX_INSTALLER)
    PKG_BUILD(  PKG_NAME        ArrayFireCPU.pkg
                INSTALL_LOCATION /usr/local/lib
                IDENTIFIER      com.arrayfire.arrayfire.cpu.lib
                PATH_TO_FILES   package/lib
                FILTERS         opencl cuda)

    PKG_BUILD(  PKG_NAME        ArrayFireCUDA.pkg
                INSTALL_LOCATION /usr/local/lib
                IDENTIFIER      com.arrayfire.arrayfire.cuda.lib
                PATH_TO_FILES   package/lib
                FILTERS         cpu opencl)

    PKG_BUILD(  PKG_NAME        ArrayFireOPENCL.pkg
                INSTALL_LOCATION /usr/local/lib
                IDENTIFIER      com.arrayfire.arrayfire.opencl.lib
                PATH_TO_FILES   package/lib
                FILTERS         cpu cuda)

    PKG_BUILD(  PKG_NAME        ArrayFireHeaders.pkg
                INSTALL_LOCATION /usr/local/include
                IDENTIFIER      com.arrayfire.arrayfire.inc
                PATH_TO_FILES   package/include)

    PKG_BUILD(  PKG_NAME        ArrayFireExtra.pkg
                INSTALL_LOCATION /usr/local/share
                IDENTIFIER      com.arrayfire.arrayfire.extra
                PATH_TO_FILES   package/share)

    PRODUCT_BUILD()
endfunction(OSX_INSTALLER)


