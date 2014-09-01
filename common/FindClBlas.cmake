# Downloads and builds the clBLAS library from github.com
# Defines the following variables
# * CLMATH_BLAS_PACKAGE_DIR     The install location of the clBLAS library
# * CLMATH_BLAS_INCLUDE_DIR     Location of the clBLAS headers
# * CLMATH_BLAS_LIBRARY_DIR     Location of the clBlas libraries
# * CLMATH_BLAS_LIBRARY         List of libraries

# Look a directory above for the clBlas folder
# TODO: Make this more robust
FIND_PATH( CLMATH_BLAS_PACKAGE_DIR
    NAMES   lib64 include bin
    PATHS   ${CMAKE_SOURCE_DIR}/../clBlas/build/package
            ${CMAKE_SOURCE_DIR}/../../clBlas/build/package)

FIND_PATH( CLMATH_BLAS_INCLUDE_DIR
    NAMES   clBLAS.h
    HINTS   ${CMAKE_SOURCE_DIR}/../clBlas/build/package/include
            ${CMAKE_SOURCE_DIR}/../../clBlas/build/package/include
    PATHS   ${CLMATH_BLAS_PACKAGE_DIR}/include
            ${CLMATH_BLAS_PACKAGE_DIR}/package/include)

FIND_PATH( CLMATH_BLAS_LIBRARY_DIR
    NAMES   libclBLAS${CMAKE_SHARED_LIBRARY_SUFFIX}
    HINTS   ${CMAKE_SOURCE_DIR}/../clBlas/build/package/lib64
            ${CMAKE_SOURCE_DIR}/../../clBlas/build/package/lib64
    PATHS   ${CLMATH_BLAS_PACKAGE_DIR}/lib64
            ${CLMATH_BLAS_PACKAGE_DIR}/package/lib64)

FIND_LIBRARY(CLMATH_BLAS_LIBRARY
    NAMES   clBLAS
    HINTS   ${CMAKE_SOURCE_DIR}/../clBlas/build/package/lib64
            ${CMAKE_SOURCE_DIR}/../../clBlas/build/package/lib64
    PATHS   ${CLMATH_BLAS_PACKAGE_DIR}/lib64
            ${CLMATH_BLAS_PACKAGE_DIR}/package/lib64)
