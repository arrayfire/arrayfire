SET(VER 415e7a0a21b7fc2cb5a6aaa7ea14a4f32e884631)
SET(URL https://github.com/kylelutz/compute/archive/${VER}.tar.gz)
SET(MD5 733daf88c39c7337c51cd5adb0efb51b)

SET(thirdPartyDir "${CMAKE_BINARY_DIR}/third_party")
SET(srcDir "${thirdPartyDir}/compute-${VER}")
SET(archive ${srcDir}.tar.gz)
SET(inflated ${srcDir}-inflated)

# the config to be used in the code
SET(boostcompute_include_dirs "${srcDir}/include")

# do we have to do it again?
SET(doExtraction ON)
IF(EXISTS "${inflated}")
    FILE(READ "${inflated}" extractedMD5)
    IF("${extractedMD5}" STREQUAL "${MD5}")
        # nope, everything looks fine
        return()
    ENDIF()
ENDIF()

# lets get and extract boost compute

MESSAGE(STATUS "BoostCompute...")
IF(EXISTS "${archive}")
    FILE(MD5 "${archive}" md5)
    IF(NOT "${md5}" STREQUAL "${MD5}")
        MESSAGE("  wrong check sum ${md5}, redownloading")
        FILE(REMOVE "${archive}")
    ENDIF()
ENDIF()

IF(NOT EXISTS "${archive}")
    MESSAGE(STATUS "  getting ${URL}")
    FILE(DOWNLOAD "${URL}" ${archive}
        STATUS rv
        SHOW_PROGRESS)
ENDIF()

MESSAGE(STATUS "  validating ${archive}")
FILE(MD5 "${archive}" md5)
IF(NOT "${md5}" STREQUAL "${MD5}")
    MESSAGE(WARNING "${archive}: Invalid check sum ${md5}. Expected was ${MD5}")
    IF("${md5}" STREQUAL "d41d8cd98f00b204e9800998ecf8427e")
        MESSAGE(STATUS "Trying wget ${URL}")
        EXECUTE_PROCESS(COMMAND wget -O ${archive} ${URL})
        FILE(MD5 "${archive}" md5_)
        IF(NOT "${md5_}" STREQUAL "${MD5}")
            MESSAGE(FATAL_ERROR "${archive}: Invalid check sum ${md5_}. Expected was ${MD5}")
        ENDIF(NOT "${md5_}" STREQUAL "${MD5}")
        MESSAGE(STATUS "wget successful")
    ENDIF("${md5}" STREQUAL "d41d8cd98f00b204e9800998ecf8427e")
ENDIF()

IF(IS_DIRECTORY ${srcDir})
    MESSAGE(STATUS "  cleaning ${cleaning}")
    FILE(REMOVE_RECURSE ${srcDir})
ENDIF()

MESSAGE(STATUS "  extracting ${archive}")
FILE(MAKE_DIRECTORY ${srcDir})
EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E tar xfz ${archive}
    WORKING_DIRECTORY ${thirdPartyDir}
    RESULT_VARIABLE rv)
IF(NOT rv EQUAL 0)
    MESSAGE(FATAL_ERROR "'${archive}' extraction failed")
ENDIF()

FILE(WRITE ${inflated} "${MD5}")
