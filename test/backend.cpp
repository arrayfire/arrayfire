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

using af::dtype_traits;
using af::getAvailableBackends;
using af::setBackend;
using std::string;
using std::vector;

const char *getActiveBackendString(af_backend active) {
    switch (active) {
        case AF_BACKEND_CPU: return "AF_BACKEND_CPU";
        case AF_BACKEND_CUDA: return "AF_BACKEND_CUDA";
        case AF_BACKEND_OPENCL: return "AF_BACKEND_OPENCL";
        default: return "AF_BACKEND_DEFAULT";
    }
}

template <typename T>
void testFunction() {
    af_info();

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

void backendTest() {
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
}

TEST(BACKEND_TEST, Basic) { backendTest(); }
