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
    af::info();

    af_array outArray = 0;
    af::dim4 dims(32, 32, 1, 1);
    ASSERT_EQ(AF_SUCCESS, af_randu(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    // cleanup
    if(outArray != 0) {
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    }
}

void infoTest()
{
    const char* ENV = getenv("AF_MULTI_GPU_TESTS");
    if(ENV && ENV[0] == '0') {
        testFunction<float>();
    } else {
        int nDevices = 0;
        ASSERT_EQ(AF_SUCCESS, af_get_device_count(&nDevices));

        int oldDevice = af::getDevice();
        for(int d = 0; d < nDevices; d++) {
            af::setDevice(d);
            testFunction<float>();
        }
        af::setDevice(oldDevice);
    }
}

TEST(Info, All)
{
    infoTest();
}
