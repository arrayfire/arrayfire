# Doxygen is required for the documentation to be built. Do not fail silently.
FIND_PACKAGE(Doxygen REQUIRED)

SET(AF_DOCS_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/doxygen.mk")
SET(AF_DOCS_CONFIG_OUT "${CMAKE_CURRENT_BINARY_DIR}/doxygen.mk.out")

SET(AF_DOCS_LAYOUT "${CMAKE_CURRENT_SOURCE_DIR}/layout.xml")
SET(AF_DOCS_LAYOUT_OUT "${CMAKE_CURRENT_BINARY_DIR}/layout.xml.out")

SET(DOCS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
SET(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets")
SET(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
SET(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")
SET(SNIPPETS_DIR "${PROJECT_SOURCE_DIR}/test")
CONFIGURE_FILE(${AF_DOCS_CONFIG} ${AF_DOCS_CONFIG_OUT})
CONFIGURE_FILE(${AF_DOCS_LAYOUT} ${AF_DOCS_LAYOUT_OUT})

###########################################################
## This generates a list of the examples cpp files and
## creates a dox file under docs/details/examples.dox
## This is used to generate documentation for examples
###########################################################
FILE(GLOB EXAMPLES_CPP
    "${EXAMPLES_DIR}/*/*.cpp")

# Sort alphabetically
# Note: example directories will be major sort order
LIST(SORT EXAMPLES_CPP)

# Get filenames and write to a string
FOREACH(SRC ${EXAMPLES_CPP})
    GET_FILENAME_COMPONENT(DIR_PATH ${SRC} DIRECTORY)
    GET_FILENAME_COMPONENT(DIR_NAME ${DIR_PATH} NAME)
    GET_FILENAME_COMPONENT(SRC_NAME ${SRC} NAME)
    SET(EXAMPLES_LIST "${EXAMPLES_LIST}\\example ${DIR_NAME}/${SRC_NAME}\n")
ENDFOREACH(SRC ${EXAMPLES_CPP})

# Write string containing file names to examples.dox
CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/CMakeModules/examples.dox.in
    ${DOCS_DIR}/details/examples.dox
)
###########################################################

ADD_CUSTOM_TARGET(docs
    ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${AF_DOCS_CONFIG_OUT}
    COMMAND cmake -E copy_directory ${ASSETS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/html
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Documentation"
    VERBATIM)

# Install Doxygen documentation
INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${AF_INSTALL_DOC_DIR}
    COMPONENT documentation
    PATTERN ".git" EXCLUDE
)

# Install man pages
#INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/man DESTINATION ${AF_INSTALL_MAN_DIR}
#    COMPONENT documentation
#)
