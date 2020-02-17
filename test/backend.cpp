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

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <af/device.h>

using af::dtype_traits;
using af::getAvailableBackends;
using af::setBackend;
using std::string;
using std::vector;

const char* getActiveBackendString(af_backend active) {
    switch (active) {
        case AF_BACKEND_CPU: return "AF_BACKEND_CPU";
        case AF_BACKEND_CUDA: return "AF_BACKEND_CUDA";
        case AF_BACKEND_OPENCL: return "AF_BACKEND_OPENCL";
        default: return "AF_BACKEND_DEFAULT";
    }
}

void testFunction(af_backend expected) {
    af_backend activeBackend = (af_backend)0;
    af_get_active_backend(&activeBackend);

    ASSERT_EQ(expected, activeBackend);

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

void backendTest() {
    int backends = getAvailableBackends();

    ASSERT_NE(backends, 0);

    bool cpu    = backends & AF_BACKEND_CPU;
    bool cuda   = backends & AF_BACKEND_CUDA;
    bool opencl = backends & AF_BACKEND_OPENCL;

    if (cpu) {
        setBackend(AF_BACKEND_CPU);
        testFunction(AF_BACKEND_CPU);
    }

    if (cuda) {
        setBackend(AF_BACKEND_CUDA);
        testFunction(AF_BACKEND_CUDA);
    }

    if (opencl) {
        setBackend(AF_BACKEND_OPENCL);
        testFunction(AF_BACKEND_OPENCL);
    }
}

TEST(BACKEND_TEST, Basic) { backendTest(); }

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
