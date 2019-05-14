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

using af::addBackendLibrary;
using af::array;
using af::Backend;
using af::dtype_traits;
using af::exception;
using af::getAvailableBackends;
using af::getBackendCount;
using af::randu;
using af::setBackend;
using af::setBackendLibrary;
using af::transpose;
using std::string;
using std::vector;

#if defined(OS_WIN)
string library_suffix = ".dll";
string libraryPrefix = "";
#elif defined(__APPLE__)
string library_suffix = ".dylib";
string libraryPrefix = "lib";
#elif defined(OS_LNX)
string library_suffix = ".so";
string libraryPrefix = "lib";
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

    printf("Active Backend Enum = %s\n", getActiveBackendString(activeBackend));

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
                    BUILD_DIR "/src/backend/cpu/libafcpu.3" + library_suffix;
                addBackendLibrary(lib_path.c_str());
                setBackendLibrary(0);
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
                    BUILD_DIR "/src/backend/cuda/libafcuda.3" + library_suffix;
                addBackendLibrary(lib_path.c_str());
                setBackendLibrary(0);
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
                    BUILD_DIR "/src/backend/opencl/libafopencl.3" + library_suffix;
                addBackendLibrary(lib_path.c_str());
                setBackendLibrary(0);
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

            string cpu_path    = BUILD_DIR "/src/backend/cpu/libafcpu" + library_suffix;
            string cuda_path   = BUILD_DIR "/src/backend/cuda/libafcuda" + library_suffix;
            string opencl_path = BUILD_DIR "/src/backend/opencl/libafopencl" + library_suffix;

            int num_backends = getBackendCount();
            ASSERT_GT(num_backends, 0);
            if (num_backends > 1) {
                string lib_path0 = cpu ? cpu_path : opencl_path;
                string lib_path1 = cuda ? cuda_path : opencl_path;
                printf("Using %s and %s\n",
                       lib_path0.c_str(), lib_path1.c_str());

                addBackendLibrary(lib_path0.c_str());
                addBackendLibrary(lib_path1.c_str());

                setBackendLibrary(0);
                array a = randu(3, 2);
                array at = transpose(a);

                setBackendLibrary(1);
                array b = randu(3, 2);

                setBackendLibrary(0);
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
            ASSERT_THROW(addBackendLibrary("qwerty.so"), exception);
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
            ASSERT_THROW(setBackendLibrary(0), exception);
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
            ASSERT_THROW(setBackendLibrary(999), exception);
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
