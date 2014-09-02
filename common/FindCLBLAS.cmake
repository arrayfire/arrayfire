# Downloads and builds the clBLAS library from github.com
# Defines the following variables
# * CLBLAS_PACKAGE_DIR     The install location of the clBLAS library
# * CLBLAS_INCLUDE_DIR     Location of the clBLAS headers
# * CLBLAS_LIBRARY_DIR     Location of the clBlas libraries
# * CLBLAS_LIBRARY         List of libraries

# Look a directory above for the clBlas folder
FIND_PATH(          CLBLAS_SOURCE_DIR
    NAMES           src/clBlas.h src/clAmdBlas.h
    PATH_SUFFIXES   clblas clBLAS clBlas CLBLAS
    DOC             "Location of the clBLAS source directory"
    PATHS           ${CMAKE_SOURCE_DIR}/..
                    ${CMAKE_SOURCE_DIR}/../..)

FIND_PATH( CLBLAS_PACKAGE_DIR
    NAMES   lib64 include bin
    DOC     "Location of the clBLAS install directory."
    PATHS   ${CLBLAS_SOURCE_DIR}/build/package
    NO_DEFAULT_PATH)

FIND_PATH( CLBLAS_INCLUDE_DIR
    NAMES   clBLAS.h
    DOC     "Location of the clBLAS include directory."
    PATHS   ${CLBLAS_PACKAGE_DIR}/include
            ${CLBLAS_PACKAGE_DIR}/package/include)

FIND_PATH( CLBLAS_LIBRARY_DIR
    NAMES   libclBLAS${CMAKE_SHARED_LIBRARY_SUFFIX}
    DOC     "Location of the clBLAS library"
    PATHS   ${CLBLAS_PACKAGE_DIR}/lib64
            ${CLBLAS_PACKAGE_DIR}/package/lib64)

FIND_LIBRARY(CLBLAS_LIBRARIES
    NAMES   clBLAS
    DOC     "Library files"
    PATHS   ${CLBLAS_PACKAGE_DIR}/lib64
            ${CLBLAS_PACKAGE_DIR}/package/lib64)

IF(CLBLAS_INCLUDE_DIR AND CLBLAS_LIBRARIES AND CLBLAS_LIBRARY_DIR)
    SET( CLBLAS_FOUND ON CACHE BOOL "CLBLAS Found" )
ELSE()
    SET( CLBLAS_FOUND OFF CACHE BOOL "CLBLAS Found" )
ENDIF()


MARK_AS_ADVANCED(
    CLBLAS_FOUND
    CLBLAS_PACKAGE_DIR
    CLBLAS_INCLUDE_DIR
    CLBLAS_LIBRARY_DIR
    CLBLAS_LIBRARIES
	)
