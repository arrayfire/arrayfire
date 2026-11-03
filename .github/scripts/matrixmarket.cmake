# Initial cache (cmake -C) for the Matrix Market tests. ci.sh sets AF_CI_MTX_FILES to what
# check-matrix-data.py printed: ON once it restored the matrices of the pinned arrayfire-data
# commit into the build tree, OFF while no commit is pinned. The tests'
# download_sparse_datasets.cmake gets each matrix through af_dep_check_and_populate, whose
# FetchContent_Populate takes FETCHCONTENT_SOURCE_DIR_<name> as the matrix's source when it
# is set, and downloads nothing. The "ci" preset in CMakePresets.json sets the same variables
# for the preset builds; check-matrix-data-test.py checks that both list the manifest's matrices.
set(AF_TEST_WITH_MTX_FILES "$ENV{AF_CI_MTX_FILES}" CACHE BOOL "" FORCE)
foreach(matrix
    JGD_Kocay/Trec4 HB/bcsstm02 HB/young4c LPnetlib/lpi_vol1 LPnetlib/lpi_qual
    Sandia/oscil_dcop_12 Sandia/oscil_dcop_42 QCD/conf6_0-4x4-20 QCD/conf6_0-4x4-30)
  string(REPLACE "/" "_" name "MTXDOWNLOAD_${matrix}")
  string(TOUPPER "${name}" name)
  set(FETCHCONTENT_SOURCE_DIR_${name} "${CMAKE_BINARY_DIR}/extern/matrixmarket/${matrix}" CACHE PATH "")
endforeach()
