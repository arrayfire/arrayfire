# TODO(umar): Document
include(CheckTypeSize)

check_type_size("int" INT_SIZE
  BUILTIN_TYPES_ONLY LANGUAGE C)

set(MKL_THREADING_LAYER "TBB")
# TODO(umar): Add the ability to select the threading layer instead of
#             defaulting to TBB
#set(MKL_THREADING_LAYER "TBB" CACHE STRING "The threading layer to choose for MKL")
#set_property(CACHE MKL_THREADING_LAYER PROPERTY STRINGS "OpenMP" "Sequential" "TBB")

find_path(MKL_INCLUDE_DIR
  NAMES
    mkl.h
    mkl_blas.h
    mkl_cblas.h
  PATHS
    /opt/intel/mkl
    $ENV{MKL_ROOT}
  PATH_SUFFIXES
    include
    )

find_path(MKL_FFTW_INCLUDE_DIR
  NAMES
    fftw3_mkl.h
  PATHS
    ${MKL_INCLUDE_DIR}/fftw)


function(find_mkl_library)
  set(options "")
  set(single_args VAR_NAME LIBRARY_NAME)
  set(multi_args "")

  cmake_parse_arguments(mkl_args "${options}" "${single_args}" "${multi_args}" ${ARGN})

  find_library(${mkl_args_VAR_NAME}
    NAMES
      ${mkl_args_LIBRARY_NAME}
    PATHS
      /opt/intel/mkl/lib
      $ENV{MKL_ROOT}/lib
    PATH_SUFFIXES
      ""
      intel64
      intel64_lin)
    #message(STATUS "${mkl_args_VAR_NAME}: ${${mkl_args_VAR_NAME}}")
endfunction()

find_mkl_library(VAR_NAME MKL_RT_LIBRARY LIBRARY_NAME mkl_rt)
find_mkl_library(VAR_NAME MKL_CORE_LIBRARY LIBRARY_NAME mkl_core)

# TODO(umar): Add the option to use the openmp threading layer for mkl
#find_mkl_library(VAR_NAME MKL_THREAD_LIBRARY LIBRARY_NAME mkl_intel_thread)
#find_mkl_library(VAR_NAME MKL_THREAD_LIBRARY LIBRARY_NAME mkl_gnu_thread)
find_mkl_library(VAR_NAME MKL_THREAD_LIBRARY LIBRARY_NAME mkl_tbb_thread)

if("${INT_SIZE}" EQUAL 4)
  find_mkl_library(VAR_NAME MKL_INTERFACE_LIBRARY LIBRARY_NAME mkl_intel_lp64)
else()
  find_mkl_library(VAR_NAME MKL_INTERFACE_LIBRARY LIBRARY_NAME mkl_intel_ilp64)
endif()

file(STRINGS "${MKL_INCLUDE_DIR}/../../../.version" MKL_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL
  REQUIRED_VARS MKL_CORE_LIBRARY MKL_INCLUDE_DIR
  VERSION_VAR MKL_VERSION
  )

if(MKL_FOUND)
  add_library(MKL::MKL SHARED IMPORTED)
  set_target_properties(MKL::MKL
    PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      IMPORTED_LOCATION "${MKL_CORE_LIBRARY}"
      INTERFACE_LINK_LIBRARIES "${MKL_INTERFACE_LIBRARY};${MKL_THREAD_LIBRARY};${CMAKE_DL_LIBS};m"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}")
endif()

 #-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
