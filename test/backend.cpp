/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

#include <af/device.h>

using std::string;
using std::vector;

template<typename T>
void testFunction()
{
    af_info();

    af_array outArray = 0;
    dim_t dims[] = {32, 32};
    ASSERT_EQ(AF_SUCCESS, af_randu(&outArray, 2, dims, (af_dtype) af::dtype_traits<T>::af_type));
    // cleanup
    if(outArray != 0) ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

void backendTest()
{
    int backends = af::getAvailableBackends();

    bool cpu    = backends & AF_BACKEND_CPU;
    bool cuda   = backends & AF_BACKEND_CUDA;
    bool opencl = backends & AF_BACKEND_OPENCL;

    if(cpu) {
        printf("\nRunning CPU Backend...\n");
        af::setBackend(AF_BACKEND_CPU);
        testFunction<float>();
    }

    if(cuda) {
        printf("\nRunning CUDA Backend...\n");
        af::setBackend(AF_BACKEND_CUDA);
        testFunction<float>();
    }

    if(opencl) {
        printf("\nRunning OpenCL Backend...\n");
        af::setBackend(AF_BACKEND_OPENCL);
        testFunction<float>();
    }
}

TEST(BACKEND_TEST, Basic)
{
    backendTest();
}
