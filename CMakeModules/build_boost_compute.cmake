set(VER 415e7a0a21b7fc2cb5a6aaa7ea14a4f32e884631)
set(URL https://github.com/kylelutz/compute/archive/${VER}.tar.gz)
set(MD5 733daf88c39c7337c51cd5adb0efb51b)

set(thirdPartyDir "${CMAKE_BINARY_DIR}/third_party")
set(srcDir "${thirdPartyDir}/compute-${VER}")
set(archive ${srcDir}.tar.gz)
set(inflated ${srcDir}-inflated)

# the config to be used in the code
set(BoostCompute_INCLUDE_DIRS "${srcDir}/include")

# do we have to do it again?
set(doExtraction ON)
if(EXISTS "${inflated}")
  file(READ "${inflated}" extractedMD5)
  if("${extractedMD5}" STREQUAL "${MD5}")
    # nope, everything looks fine
    return()
  endif()
endif()

# lets get and extract boost compute

message(STATUS "BoostCompute...")
if(EXISTS "${archive}")
  file(MD5 "${archive}" md5)
  if(NOT "${md5}" STREQUAL "${MD5}")
    message("  wrong check sum ${md5}, redownloading")
    file(REMOVE "${archive}")
  endif()
endif()

if(NOT EXISTS "${archive}")
  message(STATUS "  getting ${URL}")
  file(DOWNLOAD "${URL}" ${archive}
    STATUS rv
    SHOW_PROGRESS)
endif()

message(STATUS "  validating ${archive}")
file(MD5 "${archive}" md5)
if(NOT "${md5}" STREQUAL "${MD5}")
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
endif()

if(IS_DIRECTORY ${srcDir})
  message(STATUS "  cleaning ${cleaning}")
  file(REMOVE_RECURSE ${srcDir})
endif()

message(STATUS "  extracting ${archive}")
file(MAKE_DIRECTORY ${srcDir})
execute_process(COMMAND ${CMAKE_COMMAND} -E tar xfz ${archive}
  WORKING_DIRECTORY ${thirdPartyDir}
  RESULT_VARIABLE rv)
if(NOT rv EQUAL 0)
  message(FATAL_ERROR "'${archive}' extraction failed")
endif()

file(WRITE ${inflated} "${MD5}")
