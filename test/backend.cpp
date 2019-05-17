/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>

#include <af/device.h>

using af::array;
using af::Backend;
using af::dtype_traits;
using af::exception;
using af::getAvailableBackends;
using af::getBackendCount;
using af::randu;
using af::setBackend;
using af::transpose;
using std::string;
using std::vector;

// These paths are based on where the CI configuration puts the built libraries
const string build_dir_str = BUILD_DIR;
#if defined(_WIN32)
const string library_suffix = ".dll";
const string library_prefix_cpu = "/bin/afcpu";
const string library_prefix_cuda = "/bin/afcuda";
const string library_prefix_opencl = "/bin/afopencl";
#elif defined(__APPLE__)
const string library_suffix = ".dylib";
const string library_prefix_cpu = "/src/backend/cpu/libafcpu";
const string library_prefix_cuda = "/src/backend/cuda/libafcuda";
const string library_prefix_opencl = "/src/backend/opencl/libafopencl";
#elif defined(__linux__)
const string library_suffix = ".so";
const string library_prefix_cpu = "/src/backend/cpu/libafcpu";
const string library_prefix_cuda = "/src/backend/cuda/libafcuda";
const string library_prefix_opencl = "/src/backend/opencl/libafopencl";
#else
#error "Unsupported platform"
#endif

const char *getActiveBackendString(af_backend active) {
    switch (active) {
        case AF_BACKEND_CPU: return "AF_BACKEND_CPU";
        case AF_BACKEND_CUDA: return "AF_BACKEND_CUDA";
        case AF_BACKEND_OPENCL: return "AF_BACKEND_OPENCL";
        default: return "AF_BACKEND_DEFAULT";
    }
}

template<typename T>
void testFunction() {
    af_backend activeBackend = (af_backend)0;
    af_get_active_backend(&activeBackend);
    af_info();

    af_array outArray = 0;
    dim_t dims[]      = {32, 32};
    EXPECT_EQ(AF_SUCCESS,
              af_randu(&outArray, 2, dims, (af_dtype)dtype_traits<T>::af_type));

    // Verify backends returned by array and by function are the same
    af_backend arrayBackend = (af_backend)0;
    af_get_backend_id(&arrayBackend, outArray);
    EXPECT_EQ(arrayBackend, activeBackend);

    // cleanup
    if (outArray != 0) { ASSERT_SUCCESS(af_release_array(outArray)); }
}

TEST(BACKEND_TEST, DiffBackends) {
    EXPECT_EXIT({
            // START of actual test

            int backends = getAvailableBackends();

            ASSERT_NE(backends, 0);

            bool cpu    = backends & AF_BACKEND_CPU;
            bool cuda   = backends & AF_BACKEND_CUDA;
            bool opencl = backends & AF_BACKEND_OPENCL;

            printf("\nRunning Default Backend...\n");
            testFunction<float>();

            if (cpu) {
                printf("\nRunning CPU Backend...\n");
                setBackend(AF_BACKEND_CPU);
                testFunction<float>();
            }

            if (cuda) {
                printf("\nRunning CUDA Backend...\n");
                setBackend(AF_BACKEND_CUDA);
                testFunction<float>();
            }

            if (opencl) {
                printf("\nRunning OpenCL Backend...\n");
                setBackend(AF_BACKEND_OPENCL);
                testFunction<float>();
            }

            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            }
            else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        }, ::testing::ExitedWithCode(0), "Test succeeded");
}

TEST(BACKEND_TEST, SetCustomCpuLibrary) {
    EXPECT_EXIT({
            // START of actual test

            int backends = getAvailableBackends();
            ASSERT_NE(backends, 0);

            if (backends & AF_BACKEND_CPU) {
                string lib_path =
                    build_dir_str + library_prefix_cpu + library_suffix;
                af_add_backend_library(lib_path.c_str());
                af_set_backend_library(0);
                testFunction<float>();
            }

            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            }
            else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        }, ::testing::ExitedWithCode(0), "Test succeeded");
}

TEST(BACKEND_TEST, SetCustomCudaLibrary) {
    EXPECT_EXIT(
        {
            // START of actual test

            int backends = getAvailableBackends();
            ASSERT_NE(backends, 0);

            if (backends & AF_BACKEND_CUDA) {
                string lib_path =
                    build_dir_str + library_prefix_cuda + library_suffix;
                af_add_backend_library(lib_path.c_str());
                af_set_backend_library(0);
                testFunction<float>();
            }

            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            } else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        },
        ::testing::ExitedWithCode(0), "Test succeeded");
}

TEST(BACKEND_TEST, SetCustomOpenclLibrary) {
    EXPECT_EXIT(
        {
            // START of actual test

            int backends = getAvailableBackends();
            ASSERT_NE(backends, 0);

            if (backends & AF_BACKEND_OPENCL) {
                string lib_path =
                    build_dir_str + library_prefix_opencl + library_suffix;
                af_add_backend_library(lib_path.c_str());
                af_set_backend_library(0);
                testFunction<float>();
            }

            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            } else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        },
        ::testing::ExitedWithCode(0), "Test succeeded");
}

TEST(BACKEND_TEST, UseArrayAfterSwitchingBackends) {
    EXPECT_EXIT({
            // START of actual test

            int backends = getAvailableBackends();

            ASSERT_NE(backends, 0);

            bool cpu    = backends & AF_BACKEND_CPU;
            bool cuda   = backends & AF_BACKEND_CUDA;
            bool opencl = backends & AF_BACKEND_OPENCL;

            int num_backends = getBackendCount();
            ASSERT_GT(num_backends, 0);
            if (num_backends > 1) {
                Backend backend0 = cpu ? AF_BACKEND_CPU : AF_BACKEND_OPENCL;
                Backend backend1 = cuda ? AF_BACKEND_CUDA : AF_BACKEND_OPENCL;
                printf("Using %s and %s\n",
                       getActiveBackendString(backend0),
                       getActiveBackendString(backend1));

                setBackend(backend0);
                array a = randu(3, 2);
                array at = transpose(a);

                setBackend(backend1);
                array b = randu(3, 2);

                setBackend(backend0);
                array att = transpose(at);
                ASSERT_ARRAYS_EQ(a, att);
            }
            else {
                printf("Only 1 backend available, skipping test\n");
            }

            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            }
            else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        }, ::testing::ExitedWithCode(0), "Test succeeded");
}

TEST(BACKEND_TEST, UseArrayAfterSwitchingLibraries) {
    EXPECT_EXIT({
            // START of actual test

            int backends = getAvailableBackends();

            ASSERT_NE(backends, 0);

            bool cpu    = backends & AF_BACKEND_CPU;
            bool cuda   = backends & AF_BACKEND_CUDA;
            bool opencl = backends & AF_BACKEND_OPENCL;

            string cpu_path    = build_dir_str + library_prefix_cpu + library_suffix;
            string cuda_path   = build_dir_str + library_prefix_cuda + library_suffix;
            string opencl_path = build_dir_str + library_prefix_opencl + library_suffix;

            int num_backends = getBackendCount();
            ASSERT_GT(num_backends, 0);
            if (num_backends > 1) {
                string lib_path0 = cpu ? cpu_path : opencl_path;
                string lib_path1 = cuda ? cuda_path : opencl_path;
                printf("Using %s and %s\n",
                       lib_path0.c_str(), lib_path1.c_str());

                af_add_backend_library(lib_path0.c_str());
                af_add_backend_library(lib_path1.c_str());

                af_set_backend_library(0);
                array a = randu(3, 2);
                array at = transpose(a);

                af_set_backend_library(1);
                array b = randu(3, 2);

                af_set_backend_library(0);
                array att = transpose(at);
                ASSERT_ARRAYS_EQ(a, att);
            }
            else {
                printf("Only 1 backend available, skipping test\n");
            }

            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            }
            else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        }, ::testing::ExitedWithCode(0), "Test succeeded");
}

TEST(BACKEND_TEST, InvalidLibPath) {
    EXPECT_EXIT({
            // START of actual test
            ASSERT_EQ(af_add_backend_library("qwerty.so"), AF_ERR_LOAD_LIB);
            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            }
            else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        }, ::testing::ExitedWithCode(0), "Test succeeded");
}

TEST(BACKEND_TEST, LibIdxPointsToNullHandle) {
    EXPECT_EXIT({
            // START of actual test
            ASSERT_EQ(af_set_backend_library(0), AF_ERR_LOAD_LIB);
            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            }
            else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        }, ::testing::ExitedWithCode(0), "Test succeeded");
}

TEST(BACKEND_TEST, LibIdxExceedsMaxHandles) {
    EXPECT_EXIT({
            // START of actual test
            ASSERT_EQ(af_set_backend_library(999), AF_ERR_LOAD_LIB);
            // END of actual test

            if (HasFailure()) {
                fprintf(stderr, "Test failed");
                exit(1);
            }
            else {
                fprintf(stderr, "Test succeeded");
                exit(0);
            }
        }, ::testing::ExitedWithCode(0), "Test succeeded");
}
