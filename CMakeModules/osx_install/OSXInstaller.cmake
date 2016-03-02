#
# Builds ArrayFire Installers for OSX
#
INCLUDE(CMakeParseArguments)
INCLUDE(${CMAKE_MODULE_PATH}/Version.cmake)

SET(BIN2CPP_PROGRAM "bin2cpp")

SET(OSX_INSTALL_DIR ${CMAKE_MODULE_PATH}/osx_install)

################################################################################
## Create Directory Structure
################################################################################
SET(OSX_TEMP "${CMAKE_BINARY_DIR}/osx_install_files")

# Common files - libforge, ArrayFireConfig*.cmake
FILE(GLOB COMMONLIB "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}/libforge*.dylib")
FILE(GLOB COMMONCMAKE "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_CMAKE_DIR}/ArrayFireConfig*.cmake")

ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_COMMON)
FOREACH(SRC ${COMMONLIB} ${COMMONCMAKE})
    FILE(RELATIVE_PATH SRC_REL ${CMAKE_INSTALL_PREFIX} ${SRC})
    ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_COMMON PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       ${SRC} "${OSX_TEMP}/common/${SRC_REL}"
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       COMMENT "Copying Common files to temporary OSX Install Dir"
                       )
ENDFOREACH()

# Backends - CPU, CUDA, OpenCL, Unified
MACRO(OSX_INSTALL_SETUP BACKEND LIB)
    FILE(GLOB ${BACKEND}LIB "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}/lib${LIB}*.dylib")
    FILE(GLOB ${BACKEND}CMAKE "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_CMAKE_DIR}/ArrayFire${BACKEND}*.cmake")

    ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_${BACKEND})
    FOREACH(SRC ${${BACKEND}LIB} ${${BACKEND}CMAKE})
        FILE(RELATIVE_PATH SRC_REL ${CMAKE_INSTALL_PREFIX} ${SRC})
        ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_${BACKEND} PRE_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy
                           ${SRC} "${OSX_TEMP}/${BACKEND}/${SRC_REL}"
                           WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                           COMMENT "Copying ${BACKEND} files to temporary OSX Install Dir"
                           )
    ENDFOREACH()
ENDMACRO(OSX_INSTALL_SETUP)

OSX_INSTALL_SETUP(CPU afcpu)
OSX_INSTALL_SETUP(CUDA afcuda)
OSX_INSTALL_SETUP(OpenCL afopencl)
OSX_INSTALL_SETUP(Unified af)

# Headers
ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_INCLUDE
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                  ${CMAKE_INSTALL_PREFIX}/include "${OSX_TEMP}/include"
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                  COMMENT "Copying header files to temporary OSX Install Dir"
                  )

# Examples
ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_EXAMPLES
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                  "${CMAKE_INSTALL_PREFIX}/share/ArrayFire/examples" "${OSX_TEMP}/examples"
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                  COMMENT "Copying examples files to temporary OSX Install Dir"
                  )

# Documentation
ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_DOC
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                  "${CMAKE_INSTALL_PREFIX}/share/ArrayFire/doc" "${OSX_TEMP}/doc"
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                  COMMENT "Copying documentation files to temporary OSX Install Dir"
                  )
################################################################################

FUNCTION(PKG_BUILD)
    CMAKE_PARSE_ARGUMENTS(ARGS "" "DEPENDS;INSTALL_LOCATION;IDENTIFIER;PATH_TO_FILES;PKG_NAME;TARGETS;SCRIPT_DIR" "FILTERS" ${ARGN})

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
            DEPENDS         OSX_INSTALL_SETUP_CPU
            TARGETS         cpu_package
            INSTALL_LOCATION /usr/local
            SCRIPT_DIR      ${OSX_INSTALL_DIR}/cpu_scripts
            IDENTIFIER      com.arrayfire.pkg.arrayfire.cpu.lib
            PATH_TO_FILES   ${OSX_TEMP}/CPU
            FILTERS         opencl cuda unified)

PKG_BUILD(  PKG_NAME        ArrayFireCUDA
            DEPENDS         OSX_INSTALL_SETUP_CUDA
            TARGETS         cuda_package
            INSTALL_LOCATION /usr/local
            SCRIPT_DIR      ${OSX_INSTALL_DIR}/cuda_scripts
            IDENTIFIER      com.arrayfire.pkg.arrayfire.cuda.lib
            PATH_TO_FILES   ${OSX_TEMP}/CUDA
            FILTERS         cpu opencl unified)

PKG_BUILD(  PKG_NAME        ArrayFireOPENCL
            DEPENDS         OSX_INSTALL_SETUP_OpenCL
            TARGETS         opencl_package
            INSTALL_LOCATION /usr/local
            IDENTIFIER      com.arrayfire.pkg.arrayfire.opencl.lib
            PATH_TO_FILES   ${OSX_TEMP}/OpenCL
            FILTERS         cpu cuda unified)

PKG_BUILD(  PKG_NAME        ArrayFireUNIFIED
            DEPENDS         OSX_INSTALL_SETUP_Unified
            TARGETS         unified_package
            INSTALL_LOCATION /usr/local
            IDENTIFIER      com.arrayfire.pkg.arrayfire.unified.lib
            PATH_TO_FILES   ${OSX_TEMP}/Unified
            FILTERS         cpu cuda opencl)

PKG_BUILD(  PKG_NAME        ArrayFireCommon
            DEPENDS         OSX_INSTALL_SETUP_COMMON
            TARGETS         common_package
            INSTALL_LOCATION /usr/local
            IDENTIFIER      com.arrayfire.pkg.arrayfire.libcommon
            PATH_TO_FILES   ${OSX_TEMP}/common
            FILTERS         cpu cuda opencl unified)

PKG_BUILD(  PKG_NAME        ArrayFireHeaders
            DEPENDS         OSX_INSTALL_SETUP_INCLUDE
            TARGETS         header_package
            INSTALL_LOCATION /usr/local/include
            IDENTIFIER      com.arrayfire.pkg.arrayfire.inc
            PATH_TO_FILES   ${OSX_TEMP}/include)

PKG_BUILD(  PKG_NAME        ArrayFireExamples
            DEPENDS         OSX_INSTALL_SETUP_EXAMPLES
            TARGETS         examples_package
            INSTALL_LOCATION /usr/local/share/ArrayFire/examples
            IDENTIFIER      com.arrayfire.pkg.arrayfire.examples
            PATH_TO_FILES   ${OSX_TEMP}/examples
            FILTERS         cmake)

PKG_BUILD(  PKG_NAME        ArrayFireDoc
            DEPENDS         OSX_INSTALL_SETUP_DOC
            TARGETS         doc_package
            INSTALL_LOCATION /usr/local/share/ArrayFire/doc
            IDENTIFIER      com.arrayfire.pkg.arrayfire.doc
            PATH_TO_FILES   ${OSX_TEMP}/doc
            FILTERS         cmake)

PRODUCT_BUILD(DEPENDS ${cpu_package} ${cuda_package} ${opencl_package} ${unified_package} ${common_package} ${header_package} ${examples_package} ${doc_package})

