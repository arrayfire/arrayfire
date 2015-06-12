#
# Builds ArrayFire Installers for OSX
#
INCLUDE(CMakeParseArguments)
INCLUDE(${CMAKE_MODULE_PATH}/Version.cmake)

SET(BIN2CPP_PROGRAM "bin2cpp")

SET(OSX_INSTALL_DIR ${CMAKE_MODULE_PATH}/osx_install)

FUNCTION(PKG_BUILD)
    CMAKE_PARSE_ARGUMENTS(ARGS "" "INSTALL_LOCATION;IDENTIFIER;PATH_TO_FILES;PKG_NAME;TARGETS;SCRIPT_DIR" "FILTERS" ${ARGN})

    FOREACH(filter ${ARGS_FILTERS})
        LIST(APPEND  FILTER_LIST --filter ${filter})
    ENDFOREACH()

    IF(ARGS_SCRIPT_DIR)
        LIST(APPEND SCRPT_DIR --scripts ${ARGS_SCRIPT_DIR})
    ENDIF(ARGS_SCRIPT_DIR)

    SET(PACKAGE_NAME "${ARGS_PKG_NAME}.pkg")
    ADD_CUSTOM_COMMAND( OUTPUT ${PACKAGE_NAME}
                        DEPENDS ${ARGS_DEPENDS}
                        COMMAND pkgbuild    --install-location  ${ARGS_INSTALL_LOCATION}
                                            --identifier        ${ARGS_IDENTIFIER}
                                            --root              ${ARGS_PATH_TO_FILES}
                                            ${SCRPT_DIR}
                                            ${FILTER_LIST}
                                            ${ARGS_PKG_NAME}.pkg
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        COMMENT "Building ${ARGS_PKG_NAME} package"
                )
    ADD_CUSTOM_TARGET(${ARGS_PKG_NAME}_installer DEPENDS ${PACKAGE_NAME})

    SET("${ARGS_TARGETS}" ${ARGS_PKG_NAME}_installer PARENT_SCOPE)
ENDFUNCTION(PKG_BUILD)

FUNCTION(PRODUCT_BUILD)
    CMAKE_PARSE_ARGUMENTS(ARGS "" "" "DEPENDS" ${ARGN})
    SET(DISTRIBUTION_FILE       "${OSX_INSTALL_DIR}/distribution.dist")
    SET(DISTRIBUTION_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/distribution.dist.out")

    SET(WELCOME_FILE       "${OSX_INSTALL_DIR}/welcome.html")
    SET(WELCOME_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/welcome.html.out")

    SET(README_FILE       "${OSX_INSTALL_DIR}/readme.html")
    SET(README_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/readme.html.out")

    SET(AF_TITLE    "ArrayFire ${AF_VERSION}")
    CONFIGURE_FILE(${DISTRIBUTION_FILE} ${DISTRIBUTION_FILE_OUT})
    CONFIGURE_FILE(${WELCOME_FILE} ${WELCOME_FILE_OUT})
    CONFIGURE_FILE(${README_FILE} ${README_FILE_OUT})

    IF(BUILD_GRAPHICS)
        SET(PACKAGE_NAME "arrayfire-${AF_VERSION}.pkg")
    ELSE(BUILD_GRAPHICS)
        SET(PACKAGE_NAME "arrayfire-no-gl-${AF_VERSION}.pkg")
    ENDIF(BUILD_GRAPHICS)

    ADD_CUSTOM_COMMAND( OUTPUT ${PACKAGE_NAME}
                        DEPENDS ${ARGS_DEPENDS}
                        COMMAND pwd
                        COMMAND productbuild    --distribution  ${DISTRIBUTION_FILE_OUT}
                                                ${PACKAGE_NAME}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        COMMENT "Creating ArrayFire.pkg OSX Installer")
    ADD_CUSTOM_TARGET(osx_installer DEPENDS ${PACKAGE_NAME})
ENDFUNCTION(PRODUCT_BUILD)


PKG_BUILD(  PKG_NAME        ArrayFireCPU
            DEPENDS         afcpu
            TARGETS         cpu_package
            INSTALL_LOCATION /usr/local/lib
            SCRIPT_DIR      ${OSX_INSTALL_DIR}/cpu_scripts
            IDENTIFIER      com.arrayfire.pkg.arrayfire.cpu.lib
            PATH_TO_FILES   package/lib
            FILTERS         opencl cuda)

PKG_BUILD(  PKG_NAME        ArrayFireCUDA
            DEPENDS         afcuda
            TARGETS         cuda_package
            INSTALL_LOCATION /usr/local/lib
            SCRIPT_DIR      ${OSX_INSTALL_DIR}/cuda_scripts
            IDENTIFIER      com.arrayfire.pkg.arrayfire.cuda.lib
            PATH_TO_FILES   package/lib
            FILTERS         cpu opencl)

PKG_BUILD(  PKG_NAME        ArrayFireOPENCL
            DEPENDS         afopencl
            TARGETS         opencl_package
            INSTALL_LOCATION /usr/local/lib
            IDENTIFIER      com.arrayfire.pkg.arrayfire.opencl.lib
            PATH_TO_FILES   package/lib
            FILTERS         cpu cuda)

PKG_BUILD(  PKG_NAME        ArrayFireHeaders
            TARGETS         header_package
            INSTALL_LOCATION /usr/local/include
            IDENTIFIER      com.arrayfire.pkg.arrayfire.inc
            PATH_TO_FILES   package/include)

PKG_BUILD(  PKG_NAME        ArrayFireExtra
            TARGETS         extra_package
            INSTALL_LOCATION /usr/local/share
            IDENTIFIER      com.arrayfire.pkg.arrayfire.extra
            PATH_TO_FILES   package/share)

PRODUCT_BUILD(DEPENDS ${cpu_package} ${cuda_package} ${opencl_package} ${header_package} ${extra_package})

