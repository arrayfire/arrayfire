#
# Builds ArrayFire Installers for OSX
#
INCLUDE(CMakeParseArguments)
INCLUDE(Version)

SET(BIN2CPP_PROGRAM "bin2cpp")

SET(OSX_INSTALL_SOURCE ${PROJECT_SOURCE_DIR}/CMakeModules/osx_install)

################################################################################
## Create Directory Structure
################################################################################
SET(OSX_TEMP "${PROJECT_BINARY_DIR}/osx_install_files")

# Common files - ArrayFireConfig*.cmake
FILE(GLOB COMMONCMAKE "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_CMAKE_DIR}/ArrayFireConfig*.cmake")

ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_COMMON)
FOREACH(SRC ${COMMONLIB} ${COMMONCMAKE})
    FILE(RELATIVE_PATH SRC_REL ${CMAKE_INSTALL_PREFIX} ${SRC})
    ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_COMMON PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       ${SRC} "${OSX_TEMP}/common/${SRC_REL}"
                       WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                       COMMENT "Copying Common files to temporary OSX Install Dir"
                       )
ENDFOREACH()

# Backends - CPU, CUDA, OpenCL, Unified
MACRO(OSX_INSTALL_SETUP BACKEND LIB)
    FILE(GLOB ${BACKEND}LIB "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}/lib${LIB}.${AF_VERSION}.dylib")
    FILE(GLOB ${BACKEND}CMAKE "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_CMAKE_DIR}/ArrayFire${BACKEND}*.cmake")

    ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_${BACKEND})
    FOREACH(SRC ${${BACKEND}LIB} ${${BACKEND}CMAKE})
        FILE(RELATIVE_PATH SRC_REL ${CMAKE_INSTALL_PREFIX} ${SRC})
        ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_${BACKEND} PRE_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy
                           ${SRC} "${OSX_TEMP}/${BACKEND}/${SRC_REL}"
                           WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                           COMMENT "Copying ${BACKEND} files to temporary OSX Install Dir - File: ${SRC_REL}"
                           )
    ENDFOREACH()
    # Create symlinks separately. Copying them in above command will do a deep copy
    ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_${BACKEND} PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E create_symlink
                       "lib${LIB}.${ArrayFire_VERSION}.dylib"
                       "lib${LIB}.${ArrayFire_VERSION_MAJOR}.dylib"
                       WORKING_DIRECTORY "${OSX_TEMP}/${BACKEND}/${AF_INSTALL_LIB_DIR}"
                       COMMENT "Copying ${BACKEND} files to temporary OSX Install Dir (Symlink)"
                       )
    ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_${BACKEND} PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E create_symlink
                       "lib${LIB}.${AF_VERSION_MAJOR}.dylib"
                       "lib${LIB}.dylib"
                       WORKING_DIRECTORY "${OSX_TEMP}/${BACKEND}/${AF_INSTALL_LIB_DIR}"
                       COMMENT "Copying ${BACKEND} files to temporary OSX Install Dir (Symlink)"
                       )
ENDMACRO(OSX_INSTALL_SETUP)

OSX_INSTALL_SETUP(CPU afcpu)
OSX_INSTALL_SETUP(CUDA afcuda)
OSX_INSTALL_SETUP(OpenCL afopencl)
OSX_INSTALL_SETUP(Unified af)

# Headers
ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_INCLUDE
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                  ${CMAKE_INSTALL_PREFIX}/include/af "${OSX_TEMP}/include/af"
                  COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_INSTALL_PREFIX}/include/arrayfire.h "${OSX_TEMP}/include/arrayfire.h"
                  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                  COMMENT "Copying header files to temporary OSX Install Dir"
                  )

# Examples
ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_EXAMPLES
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                  "${CMAKE_INSTALL_PREFIX}/share/ArrayFire/examples" "${OSX_TEMP}/examples"
                  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                  COMMENT "Copying examples files to temporary OSX Install Dir"
                  )

# Documentation
ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_DOC
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                  "${CMAKE_INSTALL_PREFIX}/share/ArrayFire/doc" "${OSX_TEMP}/doc"
                  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                  COMMENT "Copying documentation files to temporary OSX Install Dir"
                  )

IF(AF_WITH_GRAPHICS)
    MAKE_DIRECTORY("${OSX_TEMP}/Forge")

    # Forge library versions for setting up symlinks
    STRING(SUBSTRING ${FORGE_VERSION} 0 1 FORGE_VERSION_MAJOR) # Will return x

    ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_FORGE_LIB)
    SET(FORGE_LIB "${CMAKE_INSTALL_PREFIX}/lib/libforge.${FORGE_VERSION}.dylib")
    FOREACH(SRC ${FORGE_LIB})
        FILE(RELATIVE_PATH SRC_REL ${CMAKE_INSTALL_PREFIX} ${SRC})
        ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_FORGE_LIB PRE_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy
                           ${SRC} "${OSX_TEMP}/Forge/${SRC_REL}"
                           WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                           COMMENT "Copying libforge files to temporary OSX Install Dir - File: ${SRC_REL}"
        )
    ENDFOREACH()
    # Create symlinks separately. Copying them in above command will do a deep copy
    ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_FORGE_LIB PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E create_symlink
                       "libforge.${FORGE_VERSION}.dylib"
                       "libforge.${FORGE_VERSION_MAJOR}.dylib"
                       WORKING_DIRECTORY "${OSX_TEMP}/Forge/${AF_INSTALL_LIB_DIR}"
                       COMMENT "Copying libforge files to temporary OSX Install Dir (Symlink)"
                       )
    ADD_CUSTOM_COMMAND(TARGET OSX_INSTALL_SETUP_FORGE_LIB PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E create_symlink
                       "libforge.${FORGE_VERSION_MAJOR}.dylib"
                       "libforge.dylib"
                       WORKING_DIRECTORY "${OSX_TEMP}/Forge/${AF_INSTALL_LIB_DIR}"
                       COMMENT "Copying libforge files to temporary OSX Install Dir (Symlink)"
                       )

    # Forge Headers
    ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_FORGE_INCLUDE
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      "${CMAKE_INSTALL_PREFIX}/include/fg" "${OSX_TEMP}/Forge/include/fg"
                      COMMAND ${CMAKE_COMMAND} -E copy
                      "${CMAKE_INSTALL_PREFIX}/include/forge.h" "${OSX_TEMP}/Forge/include/forge.h"
                      COMMAND ${CMAKE_COMMAND} -E copy
                      "${CMAKE_INSTALL_PREFIX}/include/ComputeCopy.h" "${OSX_TEMP}/Forge/include/ComputeCopy.h"
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      COMMENT "Copying examples files to temporary OSX Install Dir"
                      )
    # Forge Examples
    ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_FORGE_EXAMPLES
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      "${CMAKE_INSTALL_PREFIX}/share/Forge/examples" "${OSX_TEMP}/Forge/examples"
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      COMMENT "Copying examples files to temporary OSX Install Dir"
                      )

    # Documentation
    ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_FORGE_DOC
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      "${CMAKE_INSTALL_PREFIX}/share/Forge/doc" "${OSX_TEMP}/Forge/doc"
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      COMMENT "Copying documentation files to temporary OSX Install Dir"
                      )

    # Forge CMake
    ADD_CUSTOM_TARGET(OSX_INSTALL_SETUP_FORGE_CMAKE
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      "${CMAKE_INSTALL_PREFIX}/share/Forge/cmake" "${OSX_TEMP}/Forge/cmake"
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      COMMENT "Copying documentation files to temporary OSX Install Dir"
                      )
ENDIF(AF_WITH_GRAPHICS)
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
    IF(AF_WITH_GRAPHICS)
        SET(DISTRIBUTION_FILE       "${OSX_INSTALL_SOURCE}/distribution.dist")
    ELSE(AF_WITH_GRAPHICS)
        SET(DISTRIBUTION_FILE       "${OSX_INSTALL_SOURCE}/distribution-no-gl.dist")
    ENDIF(AF_WITH_GRAPHICS)

    SET(DISTRIBUTION_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/distribution.dist.out")

    SET(WELCOME_FILE       "${OSX_INSTALL_SOURCE}/welcome.html")
    SET(WELCOME_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/welcome.html.out")

    SET(README_FILE       "${OSX_INSTALL_SOURCE}/readme.html")
    SET(README_FILE_OUT   "${CMAKE_CURRENT_BINARY_DIR}/readme.html.out")

    SET(AF_TITLE    "ArrayFire ${AF_VERSION}")
    CONFIGURE_FILE(${DISTRIBUTION_FILE} ${DISTRIBUTION_FILE_OUT})
    CONFIGURE_FILE(${WELCOME_FILE} ${WELCOME_FILE_OUT})
    CONFIGURE_FILE(${README_FILE} ${README_FILE_OUT})

    IF(AF_WITH_GRAPHICS)
        SET(PACKAGE_NAME "arrayfire-${AF_VERSION}.pkg")
    ELSE(AF_WITH_GRAPHICS)
        SET(PACKAGE_NAME "arrayfire-no-gl-${AF_VERSION}.pkg")
    ENDIF(AF_WITH_GRAPHICS)

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
            IDENTIFIER      com.arrayfire.pkg.arrayfire.cpu.lib
            PATH_TO_FILES   ${OSX_TEMP}/CPU
            FILTERS         opencl cuda unified)

PKG_BUILD(  PKG_NAME        ArrayFireCUDA
            DEPENDS         OSX_INSTALL_SETUP_CUDA
            TARGETS         cuda_package
            INSTALL_LOCATION /usr/local
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

IF(AF_WITH_GRAPHICS)
    PKG_BUILD(  PKG_NAME        ForgeLibrary
                DEPENDS         OSX_INSTALL_SETUP_FORGE_LIB
                TARGETS         forge_lib_package
                INSTALL_LOCATION /usr/local/lib
                SCRIPT_DIR      ${OSX_INSTALL_SOURCE}/forge_scripts
                IDENTIFIER      com.arrayfire.pkg.forge.lib
                PATH_TO_FILES   ${OSX_TEMP}/Forge/lib)

    PKG_BUILD(  PKG_NAME        ForgeHeaders
                DEPENDS         OSX_INSTALL_SETUP_FORGE_INCLUDE
                TARGETS         forge_header_package
                INSTALL_LOCATION /usr/local/include
                IDENTIFIER      com.arrayfire.pkg.forge.inc
                PATH_TO_FILES   ${OSX_TEMP}/Forge/include)

    PKG_BUILD(  PKG_NAME        ForgeExamples
                DEPENDS         OSX_INSTALL_SETUP_FORGE_EXAMPLES
                TARGETS         forge_examples_package
                INSTALL_LOCATION /usr/local/share/Forge/examples
                IDENTIFIER      com.arrayfire.pkg.forge.examples
                PATH_TO_FILES   ${OSX_TEMP}/Forge/examples
                )

    PKG_BUILD(  PKG_NAME        ForgeDoc
                DEPENDS         OSX_INSTALL_SETUP_FORGE_DOC
                TARGETS         forge_doc_package
                INSTALL_LOCATION /usr/local/share/Forge/doc
                IDENTIFIER      com.arrayfire.pkg.forge.doc
                PATH_TO_FILES   ${OSX_TEMP}/Forge/doc
                )

    PKG_BUILD(  PKG_NAME        ForgeCMake
                DEPENDS         OSX_INSTALL_SETUP_FORGE_CMAKE
                TARGETS         forge_cmake_package
                INSTALL_LOCATION /usr/local/share/Forge/cmake
                IDENTIFIER      com.arrayfire.pkg.forge.cmake
                PATH_TO_FILES   ${OSX_TEMP}/Forge/cmake
                )
ENDIF(AF_WITH_GRAPHICS)

IF(AF_WITH_GRAPHICS)
    PRODUCT_BUILD(DEPENDS ${cpu_package} ${cuda_package} ${opencl_package} ${unified_package}
                          ${common_package} ${header_package} ${examples_package} ${doc_package}
                          ${forge_lib_package} ${forge_header_package} ${forge_examples_package} ${forge_doc_package} ${forge_cmake_package}
                          )
ELSE(AF_WITH_GRAPHICS)
    PRODUCT_BUILD(DEPENDS ${cpu_package} ${cuda_package} ${opencl_package} ${unified_package}
                          ${common_package} ${header_package} ${examples_package} ${doc_package}
                          )
ENDIF(AF_WITH_GRAPHICS)

