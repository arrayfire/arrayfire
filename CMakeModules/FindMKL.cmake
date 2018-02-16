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
    IntelSWTools/compilers_and_libraries/windows/mkl/include
    )

find_path(MKL_FFTW_INCLUDE_DIR
  NAMES
    fftw3_mkl.h
  HINTS
    ${MKL_INCLUDE_DIR}/fftw)

if(WIN32)
  if(${MSVC_VERSION} GREATER_EQUAL 1900)
    set(msvc_dir "vc14")
    set(shared_suffix "_dll")
  else()
    message(WARNING "MKL: MS Version not supported for MKL")
  endif()
endif()

function(find_mkl_library)
  set(options "")
  set(single_args NAME LIBRARY_NAME)
  set(multi_args "")

  cmake_parse_arguments(mkl_args "${options}" "${single_args}" "${multi_args}" ${ARGN})

  add_library(MKL::${mkl_args_NAME}        SHARED IMPORTED)
  add_library(MKL::${mkl_args_NAME}_STATIC SHARED IMPORTED)
  find_library(${mkl_args_NAME}_LINK_LIBRARY
    NAMES
      ${mkl_args_LIBRARY_NAME}${shared_suffix}
      ${mkl_args_LIBRARY_NAME}
    PATHS
      /opt/intel/mkl/lib
      /opt/intel/tbb/lib
      /opt/intel/lib
      $ENV{MKL_ROOT}/lib
    PATH_SUFFIXES
      IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64
      IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64
      IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/${msvc_dir}
      ""
      intel64
      intel64_lin)

  message(STATUS "NAME: ${mkl_args_NAME} LIBNAME: ${mkl_args_LIBRARY_NAME} ${${mkl_args_NAME}_LINK_LIBRARY}")

  find_library(${mkl_args_NAME}_STATIC_LINK_LIBRARY
    NAMES
      ${CMAKE_STATIC_LIBRARY_PREFIX}${mkl_args_LIBRARY_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
    PATHS
      /opt/intel/mkl/lib
      /opt/intel/tbb/lib
      /opt/intel/lib
      $ENV{MKL_ROOT}/lib
    PATH_SUFFIXES
      IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64
      IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64
      IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/${msvc_dir}
      ""
      intel64
      intel64_lin)

  set_target_properties(MKL::${mkl_args_NAME}
    PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGE "C"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")

    if(WIN32)
      find_file(${mkl_args_NAME}_DLL_LIBRARY
        NAMES
          ${CMAKE_SHARED_LIBRARY_PREFIX}${mkl_args_LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}
        PATH_SUFFIXES
          IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl
          IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler
          IntelSWTools/compilers_and_libraries/windows/redist/intel64/tbb/${msvc_dir}
        NO_SYSTEM_ENVIRONMENT_PATH)

      set_target_properties(MKL::${mkl_args_NAME}
        PROPERTIES
          IMPORTED_LOCATION "${${mkl_args_NAME}_DLL_LIBRARY}"
          IMPORTED_IMPLIB "${${mkl_args_NAME}_LINK_LIBRARY}")
    else()
      set_target_properties(MKL::${mkl_args_NAME}
        PROPERTIES
          IMPORTED_LOCATION "${${mkl_args_NAME}_LINK_LIBRARY}")
    endif()
endfunction()


find_mkl_library(NAME Core LIBRARY_NAME mkl_core)
find_mkl_library(NAME RT LIBRARY_NAME mkl_rt)

# TODO(umar): Add the option to use the openmp threading layer for mkl
#find_mkl_library(VAR_NAME MKL_THREAD_LAYER_LIBRARY LIBRARY_NAME mkl_intel_thread)
#find_mkl_library(VAR_NAME MKL_THREAD_LAYER_LIBRARY LIBRARY_NAME mkl_gnu_thread)
find_mkl_library(NAME ThreadLayer LIBRARY_NAME mkl_tbb_thread)

find_mkl_library(NAME tbb LIBRARY_NAME tbb)

if("${INT_SIZE}" EQUAL 4)
  find_mkl_library(NAME Interface LIBRARY_NAME mkl_intel_lp64)
else()
  find_mkl_library(NAME Interface LIBRARY_NAME mkl_intel_ilp64)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL
  REQUIRED_VARS MKL_INCLUDE_DIR)
if(NOT WIN32)
  find_library(M_LIB m)
endif()
if(MKL_FOUND)
  add_library(MKL::MKL INTERFACE IMPORTED)
  set_target_properties(MKL::MKL
    PROPERTIES
      INTERFACE_LINK_LIBRARIES "MKL::Core;MKL::ThreadLayer;MKL::Interface;MKL::tbb;${CMAKE_DL_LIBS};${M_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR};${MKL_FFTW_INCLUDE_DIR}")
endif()
