set(VER 0.4)
set(URL https://github.com/kylelutz/compute/archive/v${VER}.tar.gz)
set(MD5 0d881bd8e8c1729559bc9b98d6b25a3c)

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
  message(FATAL_ERROR "${archive}: invalid check sum ${md5}. Expected was ${MD5}")
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
