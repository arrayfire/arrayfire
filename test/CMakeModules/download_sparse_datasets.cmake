# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(ExternalProject)

add_custom_target(mtxDownloads)

set(URL "https://sparse.tamu.edu")
set(mtx_data_dir "${CMAKE_CURRENT_BINARY_DIR}/matrixmarket")
file(MAKE_DIRECTORY ${mtx_data_dir})

function(mtxDownload name group)
  set(extproj_name mtxDownload-${group}-${name})
  set(path_prefix "${ArrayFire_BINARY_DIR}/mtx_datasets/${group}")
  ExternalProject_Add(
      ${extproj_name}
      PREFIX "${path_prefix}"
      URL "${URL}/MM/${group}/${name}.tar.gz"
      SOURCE_DIR "${mtx_data_dir}/${group}/${name}"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
    )
  add_dependencies(mtxDownloads mtxDownload-${group}-${name})
endfunction()

# Following files are used for testing mtx read fn
# integer data
mtxDownload("Trec4" "JGD_Kocay")
# real data
mtxDownload("bcsstm02" "HB")
# complex data
mtxDownload("young4c" "HB")

#Following files are used for sparse-sparse arith
# real data
#linear programming problem
mtxDownload("lpi_vol1" "LPnetlib")
mtxDownload("lpi_qual" "LPnetlib")
#Subsequent Circuit Simulation problem
mtxDownload("oscil_dcop_12" "Sandia")
mtxDownload("oscil_dcop_42" "Sandia")

# complex data
#Quantum Chemistry problem
mtxDownload("conf6_0-4x4-20" "QCD")
mtxDownload("conf6_0-4x4-30" "QCD")
