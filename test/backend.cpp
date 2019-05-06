/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/backend.h>
#include <af/data.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

using af::array;
using af::Backend;
using af::dtype_traits;
using af::exception;
using af::getActiveBackend;
using af::getAvailableBackends;
using af::getBackendCount;
using af::randu;
using af::setBackend;
using af::transpose;
using std::string;
using std::vector;

// These paths are based on where the CMake configuration puts the built
// libraries
const char* BUILD_CPU_LIB_PATH    = CPU_LIB_PATH;
const char* BUILD_CUDA_LIB_PATH   = CUDA_LIB_PATH;
const char* BUILD_OPENCL_LIB_PATH = OPENCL_LIB_PATH;

const char* getActiveBackendString(af_backend active) {
    switch (active) {
        case AF_BACKEND_CPU: return "AF_BACKEND_CPU";
        case AF_BACKEND_CUDA: return "AF_BACKEND_CUDA";
        case AF_BACKEND_OPENCL: return "AF_BACKEND_OPENCL";
        default: return "AF_BACKEND_DEFAULT";
    }
}

void testFunction() {
    af_backend activeBackend = (af_backend)0;
    af_get_active_backend(&activeBackend);
    af_info();

    ASSERT_NE(0, activeBackend);

    af_array outArray = 0;
    dim_t dims[]      = {32, 32};
    EXPECT_EQ(AF_SUCCESS, af_randu(&outArray, 2, dims, f32));

    // Verify backends returned by array and by function are the same
    af_backend arrayBackend = (af_backend)0;
    af_get_backend_id(&arrayBackend, outArray);
    EXPECT_EQ(arrayBackend, activeBackend);

    // cleanup
    if (outArray != 0) { ASSERT_SUCCESS(af_release_array(outArray)); }
}

TEST(DefaultLibPath, SetBackendDefault) {
    EXPECT_EXIT(
        {
            // START of actual test
            testFunction();
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

TEST(DefaultLibPath, SetBackendCpu) {
    EXPECT_EXIT(
        {
            // START of actual test

            int backends = getAvailableBackends();
            EXPECT_NE(backends, 0);

            if (backends & AF_BACKEND_CPU) {
                setBackend(AF_BACKEND_CPU);
                testFunction();
            } else {
                printf("CPU backend not available, skipping test\n");
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

TEST(DefaultLibPath, SetBackendCuda) {
    EXPECT_EXIT(
        {
            // START of actual test

            int backends = getAvailableBackends();
            EXPECT_NE(backends, 0);

            if (backends & AF_BACKEND_CUDA) {
                setBackend(AF_BACKEND_CUDA);
                testFunction();
            } else {
                printf("CUDA backend not available, skipping test\n");
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

TEST(DefaultLibPath, SetBackendOpencl) {
    EXPECT_EXIT(
        {
            // START of actual test

            int backends = getAvailableBackends();
            EXPECT_NE(backends, 0);

            if (backends & AF_BACKEND_OPENCL) {
                setBackend(AF_BACKEND_OPENCL);
                testFunction();
            } else {
                printf("OpenCL backend not available, skipping test\n");
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

TEST(DefaultLibPath, UseArrayAfterSwitchingBackends) {
    EXPECT_EXIT(
        {
            // START of actual test

            int backends = getAvailableBackends();

            EXPECT_NE(backends, 0);

            bool cpu    = backends & AF_BACKEND_CPU;
            bool cuda   = backends & AF_BACKEND_CUDA;
            bool opencl = backends & AF_BACKEND_OPENCL;

            int num_backends = getBackendCount();
            EXPECT_GT(num_backends, 0);
            if (num_backends > 1) {
                Backend backend0 = cpu ? AF_BACKEND_CPU : AF_BACKEND_OPENCL;
                Backend backend1 = cuda ? AF_BACKEND_CUDA : AF_BACKEND_OPENCL;
                printf("Using %s and %s\n", getActiveBackendString(backend0),
                       getActiveBackendString(backend1));

                setBackend(backend0);
                array a      = randu(3, 2);
                array a_copy = a;
                a            = transpose(a);

                setBackend(backend1);
                array b = randu(3, 2);
                b.eval();

                setBackend(backend0);
                a = transpose(a);
                ASSERT_ARRAYS_EQ(a_copy, a);
            } else {
                printf("Only 1 backend available, skipping test\n");
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

TEST(CustomLibPath, SetCustomCpuLibrary) {
    EXPECT_EXIT(
        {
            // START of actual test

            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                int backends = getAvailableBackends();
                EXPECT_NE(backends, 0);

                if (backends & AF_BACKEND_CPU) {
                    ASSERT_SUCCESS(af_add_backend_library(BUILD_CPU_LIB_PATH));
                    ASSERT_SUCCESS(af_set_backend_library(0));
                    testFunction();
                } else {
                    printf("CPU backend not available, skipping test\n");
                }
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, SetCustomCudaLibrary) {
    EXPECT_EXIT(
        {
            // START of actual test

            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                int backends = getAvailableBackends();
                EXPECT_NE(backends, 0);

                if (backends & AF_BACKEND_CUDA) {
                    ASSERT_SUCCESS(af_add_backend_library(BUILD_CUDA_LIB_PATH));
                    ASSERT_SUCCESS(af_set_backend_library(0));
                    testFunction();
                } else {
                    printf("CUDA backend not available, skipping test\n");
                }
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, SetCustomOpenclLibrary) {
    EXPECT_EXIT(
        {
            // START of actual test

            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                int backends = getAvailableBackends();
                EXPECT_NE(backends, 0);

                if (backends & AF_BACKEND_OPENCL) {
                    ASSERT_SUCCESS(
                        af_add_backend_library(BUILD_OPENCL_LIB_PATH));
                    ASSERT_SUCCESS(af_set_backend_library(0));
                    testFunction();
                } else {
                    printf("OpenCL backend not available, skipping test\n");
                }
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, UseArrayAfterSwitchingLibraries) {
    EXPECT_EXIT(
        {
            // START of actual test

            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                int backends = getAvailableBackends();

                EXPECT_NE(backends, 0);

                bool cpu    = backends & AF_BACKEND_CPU;
                bool cuda   = backends & AF_BACKEND_CUDA;
                bool opencl = backends & AF_BACKEND_OPENCL;

                int num_backends = getBackendCount();
                EXPECT_GT(num_backends, 0);
                if (num_backends > 1) {
                    string lib_path0 =
                        cpu ? BUILD_CPU_LIB_PATH : BUILD_OPENCL_LIB_PATH;
                    string lib_path1 =
                        cuda ? BUILD_CUDA_LIB_PATH : BUILD_OPENCL_LIB_PATH;
                    printf("Using %s and %s\n", lib_path0.c_str(),
                           lib_path1.c_str());

                    ASSERT_SUCCESS(af_add_backend_library(lib_path0.c_str()));
                    ASSERT_SUCCESS(af_set_backend_library(0));
                    array a      = randu(3, 2);
                    array a_copy = a;
                    a            = transpose(a);

                    ASSERT_SUCCESS(af_add_backend_library(lib_path1.c_str()));
                    ASSERT_SUCCESS(af_set_backend_library(1));
                    array b = randu(3, 2);
                    b.eval();

                    ASSERT_SUCCESS(af_set_backend_library(0));
                    a = transpose(a);
                    ASSERT_ARRAYS_EQ(a_copy, a);
                } else {
                    printf("Only 1 backend available, skipping test\n");
                }
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, UseArrayAfterSwitchingToSameLibrary) {
    EXPECT_EXIT(
        {
            // START of actual test

            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                int backends = getAvailableBackends();

                EXPECT_NE(backends, 0);

                bool cpu    = backends & AF_BACKEND_CPU;
                bool cuda   = backends & AF_BACKEND_CUDA;
                bool opencl = backends & AF_BACKEND_OPENCL;

                string custom_lib_path;
                setBackend(AF_BACKEND_DEFAULT);
                Backend default_backend = getActiveBackend();
                switch (default_backend) {
                    case AF_BACKEND_CPU:
                        custom_lib_path = BUILD_CPU_LIB_PATH;
                        break;
                    case AF_BACKEND_CUDA:
                        custom_lib_path = BUILD_CUDA_LIB_PATH;
                        break;
                    case AF_BACKEND_OPENCL:
                        custom_lib_path = BUILD_OPENCL_LIB_PATH;
                        break;
                    default:
                        fprintf(stderr, "Cannot get default backend");
                        break;
                }
                EXPECT_TRUE(default_backend == AF_BACKEND_CPU ||
                            default_backend == AF_BACKEND_CUDA ||
                            default_backend == AF_BACKEND_OPENCL);

                printf("Using %s\n", custom_lib_path.c_str());

                ASSERT_SUCCESS(af_add_backend_library(custom_lib_path.c_str()));
                ASSERT_SUCCESS(af_set_backend_library(0));
                array a      = randu(3, 3);
                array a_copy = a;
                a            = transpose(a);

                ASSERT_SUCCESS(af_add_backend_library(custom_lib_path.c_str()));
                ASSERT_SUCCESS(af_set_backend_library(1));
                a = transpose(a);
                ASSERT_ARRAYS_EQ(a_copy, a);
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, ExceedLibHandleListCapacity) {
    EXPECT_EXIT(
        {
            // START of actual test

            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                int backends = getAvailableBackends();

                EXPECT_NE(backends, 0);

                bool cpu    = backends & AF_BACKEND_CPU;
                bool cuda   = backends & AF_BACKEND_CUDA;
                bool opencl = backends & AF_BACKEND_OPENCL;

                string custom_lib_path;
                setBackend(AF_BACKEND_DEFAULT);
                Backend default_backend = getActiveBackend();
                switch (default_backend) {
                    case AF_BACKEND_CPU:
                        custom_lib_path = BUILD_CPU_LIB_PATH;
                        break;
                    case AF_BACKEND_CUDA:
                        custom_lib_path = BUILD_CUDA_LIB_PATH;
                        break;
                    case AF_BACKEND_OPENCL:
                        custom_lib_path = BUILD_OPENCL_LIB_PATH;
                        break;
                    default:
                        fprintf(stderr, "Cannot get default backend");
                        break;
                }
                EXPECT_TRUE(default_backend == AF_BACKEND_CPU ||
                            default_backend == AF_BACKEND_CUDA ||
                            default_backend == AF_BACKEND_OPENCL);

                printf("Using %s\n", custom_lib_path.c_str());

                int max_custom_bknd_libs = 7;
                for (int i = 0; i < max_custom_bknd_libs; ++i) {
                    ASSERT_SUCCESS(
                        af_add_backend_library(custom_lib_path.c_str()));
                }
                EXPECT_EQ(AF_ERR_BKND_LIB_LIST_FULL,
                          af_add_backend_library(custom_lib_path.c_str()));
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, InvalidLibPath) {
    EXPECT_EXIT(
        {
            // START of actual test
            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                EXPECT_EQ(AF_ERR_LOAD_LIB, af_add_backend_library("qwerty.so"));
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, LibIdxPointsToNullHandle) {
    EXPECT_EXIT(
        {
            // START of actual test
            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                EXPECT_EQ(AF_ERR_NO_TGT_BKND_LIB, af_set_backend_library(0));
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, LibIdxExceedsMaxHandles) {
    EXPECT_EXIT(
        {
            // START of actual test
            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (is_unified_backend) {
                EXPECT_EQ(AF_ERR_BKND_LIB_IDX_INVALID,
                          af_set_backend_library(999));
            } else {
                printf("Not using unified backend, skipping test\n");
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

TEST(CustomLibPath, AddBackendLibraryNotUnified) {
    EXPECT_EXIT(
        {
            // START of actual test
            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (!is_unified_backend) {
                ASSERT_SUCCESS(af_add_backend_library(""));
            } else {
                printf("Using unified backend, skipping test\n");
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

TEST(CustomLibPath, SetBackendLibraryNotUnified) {
    EXPECT_EXIT(
        {
            // START of actual test
            bool is_unified_backend = false;
            ASSERT_SUCCESS(af_check_unified_backend(&is_unified_backend));
            if (!is_unified_backend) {
                ASSERT_SUCCESS(af_set_backend_library(0));
            } else {
                printf("Using unified backend, skipping test\n");
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

using af::getActiveBackend;

void test_backend(std::atomic<int>& counter, int ntests,
                  af::Backend default_backend, af::Backend test_backend) {
    auto ta_backend = getActiveBackend();
    ASSERT_EQ(default_backend, ta_backend);

    // Wait until all threads reach this point
    counter++;
    while (counter < ntests) {}

    setBackend(test_backend);

    // Wait until all threads reach this point
    counter++;
    while (counter < 2 * ntests) {}

    ta_backend = getActiveBackend();
    ASSERT_EQ(test_backend, ta_backend);
}

TEST(Backend, Threads) {
    using std::thread;
    std::atomic<int> count(0);

    setBackend(AF_BACKEND_DEFAULT);
    auto default_backend = getActiveBackend();

    int numbk = af::getBackendCount();

    thread a, b, c;
    if (af::getAvailableBackends() & AF_BACKEND_CPU) {
        a = thread([&]() {
            test_backend(count, numbk, default_backend, AF_BACKEND_CPU);
        });
    }

    if (af::getAvailableBackends() & AF_BACKEND_OPENCL) {
        b = thread([&]() {
            test_backend(count, numbk, default_backend, AF_BACKEND_OPENCL);
        });
    }

    if (af::getAvailableBackends() & AF_BACKEND_CUDA) {
        c = thread([&]() {
            test_backend(count, numbk, default_backend, AF_BACKEND_CUDA);
        });
    }

    if (a.joinable()) a.join();
    if (b.joinable()) b.join();
    if (c.joinable()) c.join();
}
