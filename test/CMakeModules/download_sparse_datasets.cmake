# Copyright (c) 2021, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

set(URL "https://sparse.tamu.edu")

function(mtxDownload name group)
  set(root_dir ${ArrayFire_BINARY_DIR}/extern/matrixmarket)
  set(target_dir ${root_dir}/${group}/${name})
  set(mtx_name mtxDownload_${group}_${name})
  string(TOLOWER ${mtx_name} mtx_name)
  FetchContent_Declare(
    ${mtx_name}
    URL ${URL}/MM/${group}/${name}.tar.gz
  )
  af_dep_check_and_populate(${mtx_name})
  set_and_mark_depname(mtx_prefix ${mtx_name})
  if(AF_BUILD_OFFLINE)
    set_fetchcontent_src_dir(mtx_prefix "{name}.mtx file from {group} group")
  endif()
  if(NOT EXISTS "${target_dir}/${name}.mtx")
    file(MAKE_DIRECTORY ${target_dir})
    file(COPY ${${mtx_name}_SOURCE_DIR}/${name}.mtx DESTINATION ${target_dir})
  endif()
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
