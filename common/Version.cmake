#
# Make a version file that includes the ArrayFire version and git revision
#
SET(AF_VERSION "3.0")
SET(AF_VERSION_MINOR ".200")
EXECUTE_PROCESS(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

CONFIGURE_FILE(
    ${CMAKE_SOURCE_DIR}/common/version.h.in
    ${CMAKE_SOURCE_DIR}/include/af/version.h
)
