# Downloads and builds the clBLAS library from github.com
# Defines the following variables
# * CLMATH_BLAS_PACKAGE_DIR     The install location of the clBLAS library
# * CLMATH_BLAS_INCLUDE_DIR     Location of the clBLAS headers
# * CLMATH_BLAS_LIBRARY_DIR     Location of the clBlas libraries
# * CLMATH_BLAS_LIBRARY         List of libraries

# Set default ExternalProject root directory
FIND_PATH( CLMATH_BLAS_PACKAGE_DIR
    NAMES   lib64 include bin
    PATHS   ${CMAKE_SOURCE_DIR}/../clBLAS/build/package
            ${CMAKE_SOURCE_DIR}/../../clBLAS/build/package)

FIND_PATH( CLMATH_BLAS_INCLUDE_DIR
    NAMES clBLAS.h
    PATHS   ${CMAKE_SOURCE_DIR}/../clBLAS/build/package/include
            ${CMAKE_SOURCE_DIR}/../../clBLAS/build/package/include
            ${CLMATH_BLAS_PACKAGE_DIR}/include
            ${CLMATH_BLAS_PACKAGE_DIR}/package/include)

        MESSAGE(FINDING libclBLAS${CMAKE_SHARED_LIBRARY_SUFFIX})
FIND_PATH( CLMATH_BLAS_LIBRARY_DIR
    NAMES   libclBLAS${CMAKE_SHARED_LIBRARY_SUFFIX}
    PATHS   ${CMAKE_SOURCE_DIR}/../clBLAS/build/package/lib64
            ${CMAKE_SOURCE_DIR}/../../clBLAS/build/package/lib64
            ${CLMATH_BLAS_PACKAGE_DIR}/lib64
            ${CLMATH_BLAS_PACKAGE_DIR}/package/lib64)

FIND_LIBRARY(CLMATH_BLAS_LIBRARY
    NAMES   clBLAS
    PATHS   ${CMAKE_SOURCE_DIR}/../clBLAS/build/package/lib64
            ${CMAKE_SOURCE_DIR}/../../clBLAS/build/package/lib64
            ${CLMATH_BLAS_PACKAGE_DIR}/lib64
            ${CLMATH_BLAS_PACKAGE_DIR}/package/lib64)
