CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(Threads REQUIRED)
IF(CMAKE_USE_PTHREADS_INIT)
    SET(THREAD_LIB_FLAG "-pthread")
ELSE()
    SET(THREAD_LIB_FLAG ${CMAKE_THREAD_LIBS_INIT})
ENDIF()

IF(${BUILD_GTEST})
    INCLUDE(${CMAKE_SOURCE_DIR}/common/build_gtest.cmake)
    ExternalProject_Get_Property(googletest SOURCE_DIR BINARY_DIR)

    SET(GTEST_BINARY_DIR ${BINARY_DIR})
    SET(GTEST_SOURCE_DIR ${SOURCE_DIR})

    INCLUDE(${CMAKE_SOURCE_DIR}/common/GetTestData.cmake)
    ExternalProject_Get_Property(arrayfire_data SOURCE_DIR)

    SET(TESTDATA_SOURCE_DIR ${SOURCE_DIR})
ELSE()
    SET(GTEST_SOURCE_DIR "${CMAKE_BINARY_DIR}/third_party/src/googletest")
    SET(GTEST_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/src/googletest-build")

    SET(TESTDATA_SOURCE_DIR "${CMAKE_BINARY_DIR}/data/src/arrayfire_data")
ENDIF()

SET(GTEST_SRC               ${GTEST_SOURCE_DIR})
SET(GTEST_BUILD             ${GTEST_BINARY_DIR})
SET(GTEST_INCLUDE_DIRS      "${GTEST_SRC}/include")
SET(GTEST_LIBRARY_DIRS      "${GTEST_BUILD}/ReleaseLibs")

SET(GTEST_BOTH_LIBRARIES    "${GTEST_LIBRARY_DIRS}/libgtest.a"
                            "${GTEST_LIBRARY_DIRS}/libgtest_main.a")

ADD_DEFINITIONS("-D TEST_DIR=\"\\\"${TESTDATA_SOURCE_DIR}\\\"\"")

IF(${BUILD_CPU})
    ADD_TEST(BasicTests_cpu     basic_cpu)
    ADD_TEST(IndexingTests_cpu  index_cpu)
    ADD_TEST(Diff1Tests_cpu     diff1_cpu)

    ADD_EXECUTABLE(basic_cpu basic.cpp)
    ADD_EXECUTABLE(index_cpu index.cpp)
    ADD_EXECUTABLE(diff1_cpu diff1.cpp)
    IF(${BUILD_GTEST})
        ADD_DEPENDENCIES(basic_cpu googletest arrayfire_data)
        ADD_DEPENDENCIES(index_cpu googletest arrayfire_data)
        ADD_DEPENDENCIES(diff1_cpu googletest arrayfire_data)
    ENDIF()
    TARGET_LINK_LIBRARIES(basic_cpu afcpu ${THREAD_LIB_FLAG} ${GTEST_BOTH_LIBRARIES})
    TARGET_LINK_LIBRARIES(index_cpu afcpu ${THREAD_LIB_FLAG} ${GTEST_BOTH_LIBRARIES})
    TARGET_LINK_LIBRARIES(diff1_cpu afcpu ${THREAD_LIB_FLAG} ${GTEST_BOTH_LIBRARIES})
ENDIF()

IF(${BUILD_CUDA})
    ADD_TEST(BasicTests_cuda    basic_cuda)
    ADD_TEST(IndexingTests_cuda  index_cuda)

    ADD_EXECUTABLE(basic_cuda basic.cpp)
    ADD_EXECUTABLE(index_cuda index.cpp)
    IF(${BUILD_GTEST})
        ADD_DEPENDENCIES(basic_cuda googletest arrayfire_data)
        ADD_DEPENDENCIES(index_cuda googletest arrayfire_data)
    ENDIF()
    TARGET_LINK_LIBRARIES(basic_cuda afcuda ${THREAD_LIB_FLAG} ${GTEST_BOTH_LIBRARIES})
    TARGET_LINK_LIBRARIES(index_cuda afcuda ${THREAD_LIB_FLAG} ${GTEST_BOTH_LIBRARIES})
ENDIF()

IF(${BUILD_OPENCL})
    ADD_TEST(BasicTests_opencl  basic_opencl)
    ADD_TEST(IndexingTests_opencl  index_opencl)

    ADD_EXECUTABLE(basic_opencl basic.cpp)
    ADD_EXECUTABLE(index_opencl index.cpp)
    IF(${BUILD_GTEST})
        ADD_DEPENDENCIES(basic_opencl googletest arrayfire_data)
        ADD_DEPENDENCIES(index_opencl googletest arrayfire_data)
    ENDIF()
    TARGET_LINK_LIBRARIES(basic_opencl afopencl ${THREAD_LIB_FLAG} ${GTEST_BOTH_LIBRARIES})
    TARGET_LINK_LIBRARIES(index_opencl afopencl ${THREAD_LIB_FLAG} ${GTEST_BOTH_LIBRARIES})
ENDIF()

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GTEST_INCLUDE_DIRS})
