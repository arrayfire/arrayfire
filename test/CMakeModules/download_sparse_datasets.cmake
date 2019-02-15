# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

set(URL "https://sparse.tamu.edu")

function(download_mtx name group)
  if(AF_TEST_WITH_MTX_FILES)
    set(file_name "${group}/${name}.tar.gz")
    if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/matrixmarket/${file_name}")
      file(DOWNLOAD
          "${URL}/MM/${file_name}"
          ${CMAKE_CURRENT_BINARY_DIR}/matrixmarket/${file_name}
          INACTIVITY_TIMEOUT 600
          SHOW_PROGRESS
          STATUS out_status
          TLS_VERIFY ON
          )
      list(GET out_status 0 error_code)
      list(GET out_status 1 error_string)
      if (${error_code} EQUAL 0)
        message("Downloaded ${name} file from sparse.tamu.edu")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/matrixmarket/${group}")
        execute_process(
          COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_BINARY_DIR}/matrixmarket/${file_name}"
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/matrixmarket/${group}"
          )
        message("Extracted mtx files to test data directory")
      else ()
        if (${error_code} EQUAL 503)
            message(WARNING "${URL} service unavailable")
        elseif (${error_code} EQUAL 504)
            message(WARNING "Request to ${URL} timedout")
        elseif (${error_code} EQUAL 521)
          # CLOUDFLARE error code
          message(WARNING "Request to ${URL} has been refused")
        elseif (${error_code} EQUAL 523)
          # CLOUDFLARE error code
          message(WARNING "${URL} is unreachable")
        else ()
          message(WARNING  "Failed to download ${name} file from sparse.tamu.edu")
          message(WARNING  "Failure message: ${error_string}")
        endif ()
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/matrixmarket/${file_name}")

        set(AF_TEST_WITH_MTX_FILES OFF CACHE BOOL
            "Download and run tests on large matrices form sparse.tamu.edu"
            FORCE)
      endif ()
    endif ()
  endif ()
endfunction()

# Following files are used for testing mtx read fn
# integer data
download_mtx("Trec4" "JGD_Kocay")
# real data
download_mtx("bcsstm02" "HB")
# complex data
download_mtx("young4c" "HB")

#Following files are used for sparse-sparse arith
# real data
#linear programming problem
download_mtx("lpi_vol1" "LPnetlib")
download_mtx("lpi_qual" "LPnetlib")
#Subsequent Circuit Simulation problem
download_mtx("oscil_dcop_12" "Sandia")
download_mtx("oscil_dcop_42" "Sandia")

# complex data
#Quantum Chemistry problem
download_mtx("conf6_0-4x4-20" "QCD")
download_mtx("conf6_0-4x4-30" "QCD")
